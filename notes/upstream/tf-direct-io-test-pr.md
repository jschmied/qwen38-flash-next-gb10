POSTED 2026-10-07 as ashhart/TensorFold#486 (branch jschmied:fix-direct-io-test 9fc4b99).

## What this changes

`direct_io`'s test "direct reads return the same bytes as buffered reads" opened its buffered reference with `File.open`, which makes the descriptor O_DIRECT wherever the file system allows it; setting `.direct = false` afterwards does not change the descriptor. On ext4 on NVMe (a DGX Spark's root disk) the reference `pread` of 1 byte into an unaligned buffer then fails with EINVAL, and the test reports `expected 1, found -1`. Where O_DIRECT is refused (overlay, tmpfs) `File.open` falls back to the page cache and the test passes, which is why it can go unnoticed.

The reference descriptor is now opened without O_DIRECT. The direct handle is unchanged, so the test still checks the O_DIRECT path. Test only, 3 lines.

## Receipt

- Environment: `main` at 041d14a, head 9fc4b99. Zig 0.17.0, DGX Spark (GB10), the repository on the root ext4 file system.
- Before: `zig build test` fails in `direct_io` with `expected 1, found -1`, on `main` as on every branch.
- After: `zig build test`: 17 of 17 steps, every test passes.
