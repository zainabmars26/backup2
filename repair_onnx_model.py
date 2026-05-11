import onnx

model = onnx.load(r"C:\Users\Zainab.Alawneh\Desktop\tasks\drone_detection\runs\detect\train\weights\best.onnx")
graph = model.graph

# Find the no-op Tile node
noop_tile = None
for node in graph.node:
    if node.op_type == "Tile" and node.name == "/model.23/Tile":
        noop_tile = node
        break

if noop_tile is None:
    print("Node not found — check the name")
else:
    tile_input  = noop_tile.input[0]   # what feeds into it
    tile_output = noop_tile.output[0]  # what it claims to produce

    # Rewire: replace all uses of tile_output with tile_input
    for node in graph.node:
        for i, inp in enumerate(node.input):
            if inp == tile_output:
                node.input[i] = tile_input

    # Also fix graph outputs if needed
    for out in graph.output:
        if out.name == tile_output:
            out.name = tile_input

    graph.node.remove(noop_tile)
    print("Removed no-op Tile node successfully")

    onnx.checker.check_model(model)
    onnx.save(model, r"C:\Users\Zainab.Alawneh\Desktop\tasks\drone_detection\runs\detect\train\weights\best_fixed.onnx")
    print("Saved to best_fixed.onnx")
