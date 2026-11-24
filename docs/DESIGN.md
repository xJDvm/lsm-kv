# lsm-kv — design

## Problem

`lsm-kv` is an embedded, persistent key-value storage engine built on a
log-structured merge tree (LSM tree), the design behind LevelDB, RocksDB,
Cassandra and ScyllaDB. It is written in pure Python with **zero runtime
dependencies** (stdlib only).

Why it is technically interesting:

- **Durability is subtle.** A write must survive a crash at any byte. That
  needs a checksummed write-ahead log, torn-write detection, atomic metadata
  updates (`write tmp → fsync → rename → fsync dir`) and an idempotent recovery
  path.
- **Reads must merge many sorted sources** (memtable + N immutable tables) with
  correct shadowing: the newest version wins and tombstones hide older values.
- **On-disk formats must be defensive.** A corrupted or malicious file must
  raise a typed `CorruptionError`, never an `IndexError`, a huge allocation or
  a silent wrong answer.
- **Correctness is demonstrated, not claimed**: a Hypothesis stateful test
  drives the engine and a plain `dict` with the same random operations
  (including flush, compaction and reopen) and requires identical answers.

## Scope

### Night 1 (MVP)
- Skip-list memtable with tombstones.
- Write-ahead log with CRC32 framing and torn-tail recovery.
- Bloom filter (double hashing over BLAKE2b).
- SSTable writer/reader: data blocks, sparse index, bloom block, checksummed footer.
- K-way merge iterator (newest wins, tombstone shadowing).
- `DB` API: `put / get / delete / scan / flush / compact / close`, manifest, crash recovery.
- Size-tiered compaction with tombstone GC when compacting into the bottom table.
- Tests: unit, exhaustive WAL truncation/bit-flip, SSTable fuzzing, stateful model test.
- Benchmark script with numbers measured on the night, CLI to inspect SSTables, CI.

### Later nights
- Leveled compaction with per-level size targets and key-range partitioning.
- Fault-injecting filesystem shim to crash at every syscall boundary during flush/compaction.
- Block cache (LRU) and prefix compression in data blocks.
- Snapshots / MVCC via sequence numbers.
- Background flush/compaction thread.

## Architecture

```
            put/delete                       get/scan
                │                                │
                ▼                                ▼
   ┌──────────────────────┐        ┌─────────────────────────────┐
   │ WAL (append, CRC32)  │        │ MergeIterator (heapq)       │
   └──────────┬───────────┘        │  memtable > sst_n > … > sst_1│
              ▼                    └──────────────┬──────────────┘
   ┌──────────────────────┐   flush   ┌──────────▼──────────┐
   │ Memtable (skip list) │ ────────▶ │ SSTables (immutable)│ ◀─ compaction
   └──────────────────────┘           └──────────┬──────────┘
                                                 │
                                       MANIFEST (atomic JSON)
```

Package layout (`src/lsmkv/`):

| Module | Public interface |
|---|---|
| `errors.py` | `LSMError`, `CorruptionError(LSMError)`, `ClosedError(LSMError)` |
| `types.py` | `TOMBSTONE` sentinel, `Entry = tuple[bytes, bytes \| None]` (None = tombstone), limits `MAX_KEY_SIZE = 64 KiB`, `MAX_VALUE_SIZE = 16 MiB`, `validate_key/validate_value` |
| `memtable.py` | `Memtable(seed: int = 0)`: `put(k, v)`, `delete(k)`, `get(k) -> bytes \| None \| Missing`, `items(start=None, end=None) -> Iterator[Entry]`, `__len__`, `approx_bytes` |
| `wal.py` | `WAL.open(path, sync: bool)`: `append_put`, `append_delete`, `close`; `replay(path) -> Iterator[Op]` that stops at the first torn/corrupt record and `truncate_corrupt_tail(path)` |
| `bloom.py` | `BloomFilter.for_capacity(n, fp_rate)`, `add`, `might_contain`, `to_bytes`, `from_bytes` (validated) |
| `sstable.py` | `write_sstable(path, entries: Iterable[Entry], block_size=4096, fp_rate=0.01)`, `SSTableReader(path)`: `get(k)`, `items(start, end)`, `min_key`, `max_key`, `__len__`, `close` |
| `merge.py` | `merge_iterators(sources: Sequence[Iterator[Entry]]) -> Iterator[Entry]` (index 0 = newest), `drop_tombstones(it)` |
| `manifest.py` | `Manifest.load(dir)`, `Manifest.commit(dir, tables, next_id)` — atomic replace |
| `db.py` | `DB.open(path, *, memtable_bytes=4 MiB, compaction_trigger=4, sync=False)`: `put`, `get`, `delete`, `scan(start=None, end=None)`, `flush`, `compact`, `close`, context manager |
| `cli.py` / `__main__.py` | `python -m lsmkv inspect <file.sst>` |

### On-disk formats (all integers little-endian, `struct`)

**WAL record**: `crc32:u32 | len:u32 | payload`, payload = `op:u8 | klen:u32 | vlen:u32 | key | value`.
CRC covers `len + payload`. Replay stops at the first record that is short or
fails its CRC (a torn tail is the expected result of a crash mid-append).

**SSTable**:
```
[data block]* [index block] [bloom block] [footer]
data block  = entries… | crc32:u32          entry = kind:u8 | klen:u32 | vlen:u32 | key | value
index block = (klen:u32 | last_key | off:u64 | len:u32)… | crc32:u32
bloom block = m_bits:u32 | k:u8 | bitarray | crc32:u32
footer (48 B) = index_off:u64 | index_len:u32 | bloom_off:u64 | bloom_len:u32
                | count:u64 | magic "LSMKVT01" | crc32:u32 (over the previous bytes) | pad
```

**MANIFEST**: JSON `{"version": 1, "next_id": n, "tables": [ids, newest first]}`,
written as `MANIFEST.tmp → fsync → os.replace → fsync(dir)`.

### Invariants
1. Keys within a memtable, a data block, and an SSTable are strictly increasing.
2. Sources are ordered by recency; for equal keys the newest source wins.
3. A tombstone shadows every older value of its key; tombstones are dropped
   only when compacting into the oldest (bottom) table.
4. Recovery is idempotent: replaying a WAL whose contents are already in an
   SSTable listed by the manifest yields the same logical state.
5. Every length read from disk is bounds-checked against the remaining file
   size and the configured limits before allocation.
6. Files not referenced by the manifest are orphans from an interrupted
   flush/compaction and are deleted on open.

### Write / flush protocol
1. `put` → WAL append (optional `fsync`) → memtable insert.
2. When `memtable.approx_bytes >= memtable_bytes`: write `<id>.sst.tmp`, fsync,
   rename to `<id>.sst`, commit manifest with `id` at the front, then reset the WAL.
3. A crash between any two steps leaves either the old manifest (orphan `.sst`
   deleted, WAL replayed) or the new one (WAL replay is idempotent, invariant 4).

### Concurrency
Single process. A `threading.Lock` serializes mutations; documented as not
safe across processes (no file lock in night 1).

## Testing
- **Unit**: each module, edge cases (empty table, single key, key at block
  boundary, max-size key/value, empty values distinct from tombstones).
- **Exhaustive crash tests**: truncate a WAL at *every* byte offset and flip
  every bit of a small WAL; recovery must return a prefix of the written
  operations and never raise anything other than handled corruption.
- **Fuzz**: Hypothesis feeds arbitrary bytes and mutated valid SSTables to the
  reader; the only allowed exception is `CorruptionError`.
- **Stateful model test**: `RuleBasedStateMachine` with rules
  put/delete/get/scan/flush/compact/reopen against a `dict`.
- **Statistical**: bloom filter false-positive rate within 2× of target.
- **Benchmarks**: `benchmarks/bench.py` — sequential/random writes, random
  reads (hit and miss), full scan. Numbers in the README are only those
  measured on the night, with hardware and Python version stated.

## Dependencies
- Runtime: none.
- Dev (pinned in `uv.lock`): `pytest`, `hypothesis`, `ruff`, `mypy`, `pip-audit`.
  All are widely used and maintained. `uv` manages the venv (`.venv/`).

## Security considerations
- On-disk data is untrusted input: every length/offset is validated, CRCs are
  checked, sizes are capped; no `pickle`, no `eval`. CRC32 detects accidental
  corruption only — it is **not** an integrity guarantee against an attacker
  (documented in the README).
- The skip list uses `random.Random(seed)`: non-cryptographic randomness is
  fine here (balancing, not security) and is seeded for determinism.
- The manifest is parsed with `json` and schema-checked (types, ranges, no
  path components in table ids — ids are integers, file names are derived).

## Commit plan
1. `docs: add design document`
2. `chore: scaffold package, tooling and license`
3. `feat(memtable): add skip-list memtable with tombstones`
4. `feat(wal): add checksummed write-ahead log with torn-tail recovery`
5. `test(wal): exhaustively test truncation and bit-flip recovery`
6. `feat(bloom): add bloom filter with double hashing`
7. `feat(sstable): add block-based sstable writer and reader`
8. `test(sstable): fuzz reader against corrupted files`
9. `feat(merge): add k-way merge iterator with tombstone shadowing`
10. `feat(db): add DB with WAL, flush, manifest and recovery`
11. `feat(db): add range scans across memtable and sstables`
12. `feat(compaction): add size-tiered compaction with tombstone GC`
13. `test(db): add stateful model test against a dict`
14. `feat(cli): add sstable inspect command`
15. `perf: add benchmark script`
16. `ci: add GitHub Actions workflow`
17. `docs: add README with architecture and measured benchmarks`
