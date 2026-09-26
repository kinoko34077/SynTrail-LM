use std::fs;

fn gui_source() -> String {
    fs::read_to_string("src/bin/syntrail_gui.rs").expect("read syntrail_gui.rs")
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
