import unittest

from gesture_manager.runtime import (
    CommandQueue,
    RuntimeCommand,
    RuntimeController,
    RuntimeState,
)


class RuntimeControllerTests(unittest.TestCase):
    def test_transitions_are_idempotent_and_include_lifecycle_states(self) -> None:
        states: list[RuntimeState] = []
        controller = RuntimeController(
            on_change=lambda state, _error: states.append(state)
        )

        self.assertTrue(controller.start())
        self.assertEqual(controller.state, RuntimeState.ACTIVE)
        self.assertFalse(controller.start())

        self.assertTrue(controller.pause())
        self.assertEqual(controller.state, RuntimeState.PAUSED)
        self.assertFalse(controller.pause())
        self.assertTrue(controller.resume())
        self.assertEqual(controller.state, RuntimeState.ACTIVE)
        self.assertFalse(controller.resume())

        self.assertTrue(controller.stop())
        self.assertEqual(controller.state, RuntimeState.STOPPED)
        self.assertFalse(controller.stop())
        self.assertEqual(
            states,
            [
                RuntimeState.STARTING,
                RuntimeState.ACTIVE,
                RuntimeState.PAUSED,
                RuntimeState.ACTIVE,
                RuntimeState.STOPPING,
                RuntimeState.STOPPED,
            ],
        )

    def test_command_queue_drains_on_consumer_and_repeated_commands_are_safe(self) -> None:
        commands: CommandQueue[RuntimeCommand] = CommandQueue()
        controller = RuntimeController()
        commands.submit(RuntimeCommand.START)
        commands.submit(RuntimeCommand.PAUSE)
        commands.submit(RuntimeCommand.PAUSE)
        commands.submit(RuntimeCommand.RESUME)

        handled = commands.process(controller.handle_command)

        self.assertEqual(handled, 4)
        self.assertEqual(controller.state, RuntimeState.ACTIVE)
        self.assertEqual(commands.process(controller.handle_command), 0)

    def test_action_errors_are_exposed_and_stop_can_recover(self) -> None:
        def fail_to_start() -> None:
            raise OSError("camera unavailable")

        controller = RuntimeController(start_action=fail_to_start)
        self.assertTrue(controller.start())
        self.assertEqual(controller.state, RuntimeState.ERROR)
        self.assertEqual(controller.error, "OSError: camera unavailable")

        self.assertTrue(controller.stop())
        self.assertEqual(controller.state, RuntimeState.STOPPED)
        self.assertIsNone(controller.error)

    def test_start_can_retry_after_error_and_calls_runtime_action_again(self) -> None:
        attempts = 0

        def start_runtime() -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise OSError("camera busy")

        controller = RuntimeController(start_action=start_runtime)
        self.assertTrue(controller.start())
        self.assertEqual(controller.state, RuntimeState.ERROR)
        self.assertTrue(controller.start())
        self.assertEqual(attempts, 2)
        self.assertEqual(controller.state, RuntimeState.ACTIVE)
        self.assertIsNone(controller.error)


if __name__ == "__main__":
    unittest.main()
