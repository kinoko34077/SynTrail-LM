use std::fs;

fn gui_source() -> String {
    fs::read_to_string("src/bin/syntrail_gui.rs").expect("read syntrail_gui.rs")
}

fn section<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start_index = source.find(start).unwrap_or_else(|| panic!("missing section start: {start}"));
    let end_index = source[start_index + start.len()..]
        .find(end)
        .map(|offset| start_index + start.len() + offset)
        .unwrap_or_else(|| panic!("missing section end: {end}"));
    &source[start_index..end_index]
}

#[test]
fn send_dispatch_does_not_generate_on_the_egui_update_thread() {
    let source = gui_source();

    assert!(source.contains("thread::spawn") || source.contains("std::thread::spawn"),
        "chat generation must be dispatched to a background worker");
    assert!(source.contains("try_recv"),
        "the egui update loop must poll worker completion without blocking");
    assert!(!source.contains("Action::Send            => self.send_message()"),
        "Action::Send must not synchronously call the old blocking send_message path");
}

#[test]
fn generation_has_an_explicit_in_flight_state_and_duplicate_send_is_disabled() {
    let source = gui_source();

    assert!(source.contains("Generating"), "UI must expose an explicit generating state");
    assert!(source.contains("is_generating"), "generation lifecycle must be queryable");
    assert!(source.contains("add_enabled"), "Send must be disabled while a request is active");
}

#[test]
fn model_mutating_actions_are_rejected_while_generation_owns_the_handle() {
    let source = gui_source();

    assert!(source.contains("Generation in progress"),
        "New/Open/Save/feedback-style model mutations need a visible busy rejection while generation owns the model");
}

#[test]
fn invalid_or_multiple_chat_drop_reports_visible_status() {
    let source = gui_source();
    let drop_block = section(&source, "match route_drop", "let model_path_str");

    assert!(drop_block.contains("DropResult::MultipleFiles =>"));
    assert!(drop_block.contains("Drop rejected: multiple files are not supported in Chat"),
        "multiple-file D&D must explain why the drop was rejected");
    assert!(drop_block.contains("DropResult::Unsupported(path) =>"));
    assert!(drop_block.contains("Unsupported dropped file"),
        "unsupported D&D must identify the rejected file in status");
}

#[test]
fn completed_turn_commits_the_already_computed_analytics_snapshot_immediately() {
    let source = gui_source();
    let completion = section(&source, "Ok((turn_id, output)) => {", "Err(error) => {");

    assert!(completion.contains("self.analytics = a;"),
        "the analytics snapshot already computed for the completed turn must become the visible snapshot immediately");
    assert!(!source.contains("turns_since_refresh"),
        "the old stale N-turn display counter must not remain active");
    assert!(!source.contains("refresh_interval"),
        "the old N-turn display interval must not remain as a misleading UI control");
}

#[test]
fn close_during_generation_is_canceled_and_deferred_until_handle_returns() {
    let source = gui_source();

    assert!(source.contains("viewport().close_requested()"),
        "the GUI must observe native close requests before allowing the viewport to exit");
    assert!(source.contains("ViewportCommand::CancelClose"),
        "a close request during generation must be canceled");
    assert!(source.contains("close_requested_during_generation"),
        "a canceled close must be remembered until the worker returns AppHandle");
    assert!(source.contains("ViewportCommand::Close"),
        "the deferred close must continue after the normal dirty-model confirmation");
    assert!(source.contains("Finishing generation"),
        "the user must see why closing is temporarily deferred");
}
