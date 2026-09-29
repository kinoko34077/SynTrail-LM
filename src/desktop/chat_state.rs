//! Pure Chat GUI lifecycle rules shared by the GUI and behavioral tests.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChatLifecycle {
    Idle,
    Generating,
    RestartRequired,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorkerExit {
    Completed,
    Failed,
    Panicked,
    Disconnected,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CloseRequestDecision {
    Allow,
    Defer,
    Cancel,
}

pub const RESTART_REQUIRED_STATUS: &str = "Generation crashed: restart the app before continuing.";

impl ChatLifecycle {
    pub const fn is_generating(self) -> bool {
        matches!(self, Self::Generating)
    }

    pub const fn restart_required(self) -> bool {
        matches!(self, Self::RestartRequired)
    }

    pub const fn can_send(self) -> bool {
        matches!(self, Self::Idle)
    }

    pub const fn can_mutate_model(self) -> bool {
        matches!(self, Self::Idle)
    }
}

pub const fn lifecycle_after_worker_exit(exit: WorkerExit) -> ChatLifecycle {
    match exit {
        WorkerExit::Completed | WorkerExit::Failed => ChatLifecycle::Idle,
        WorkerExit::Panicked | WorkerExit::Disconnected => ChatLifecycle::RestartRequired,
    }
}

pub const fn restart_required_message(exit: WorkerExit) -> Option<&'static str> {
    match exit {
        WorkerExit::Panicked => {
            Some("Generation crashed: worker panicked. Restart the app before continuing.")
        }
        WorkerExit::Disconnected => Some(
            "Generation crashed: worker disconnected before returning the model handle. Restart the app before continuing.",
        ),
        WorkerExit::Completed | WorkerExit::Failed => None,
    }
}

pub const fn close_request_decision(
    lifecycle: ChatLifecycle,
    is_dirty: bool,
    discard_confirmed: bool,
) -> CloseRequestDecision {
    if lifecycle.is_generating() {
        CloseRequestDecision::Defer
    } else if !is_dirty || discard_confirmed {
        CloseRequestDecision::Allow
    } else {
        CloseRequestDecision::Cancel
    }
}
