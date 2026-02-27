from Vision_system.workspace_state import load_json_snapshot, state_to_api_payload

state = load_json_snapshot("positions.json")

payload_str = state_to_api_payload(
    state,
    compact_keys=True,
    drop_nulls=True,
)

print(payload_str)