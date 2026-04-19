
from pathlib import Path


def read_last_timestep(filepath):
    """Return {atom_id: spin} for the last timestep in a dump.out file."""
    atoms = {}
    col_id = col_s = -1
    reading = False

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith("ITEM: TIMESTEP"):
                atoms = {}          # reset — we only keep the last block
                reading = False
            elif line.startswith("ITEM: ATOMS"):
                headers = line.split()[2:]
                col_id, col_s = headers.index("id"), headers.index("s")
                reading = True
            elif reading and not line.startswith("ITEM:"):
                parts = line.split()
                if len(parts) > max(col_id, col_s):
                    atoms[int(parts[col_id])] = int(parts[col_s])

    return atoms


root = Path(".")
ref  = read_last_timestep(root / "serial" / "dump.out")

for impl_dir in sorted(root.iterdir()):
    dump = impl_dir / "dump.out"
    if not dump.exists() or impl_dir.name == "serial":
        continue

    other = read_last_timestep(dump)
    diffs = [aid for aid in ref if ref[aid] != other.get(aid)]

    if diffs:
        print(f"FAIL  {impl_dir.name}  --  {len(diffs)} atom(s) differ")
    else:
        print(f"PASS  {impl_dir.name}")