import unittest

from inference_server.server import build_command, load_profiles, printable_command


class ServerLauncherTests(unittest.TestCase):
    def environment(self, profile="qwen35-9b", **overrides):
        values = {
            "INFERENCE_MODEL": profile,
            "INFERENCE_API_KEY": "secret",
        }
        values.update(overrides)
        return values

    def test_registry_contains_requested_runnable_models(self):
        profiles = load_profiles()
        self.assertEqual(
            {name for name, profile in profiles.items() if profile.get("available", True)},
            {
                "qwen35-9b",
                "glm46v-flash",
                "qwen3-vl-8b-thinking",
                "internvl35-14b",
                "kimi-vl-a3b-thinking",
            },
        )

    def test_builds_multimodal_vllm_command(self):
        command = build_command(self.environment())
        self.assertEqual(command[:3], ["vllm", "serve", "Qwen/Qwen3.5-9B"])
        self.assertIn("--enable-prefix-caching", command)
        self.assertEqual(command[command.index("--reasoning-parser") + 1], "qwen3")
        self.assertEqual(command[command.index("--limit-mm-per-prompt") + 1], '{"image":8}')

    def test_builds_sglang_command(self):
        command = build_command(
            self.environment("qwen3-vl-8b-thinking", INFERENCE_BACKEND="sglang")
        )
        self.assertEqual(command[:3], ["python3", "-m", "sglang.launch_server"])
        self.assertEqual(command[command.index("--model-path") + 1], "Qwen/Qwen3-VL-8B-Thinking")
        self.assertEqual(command[command.index("--reasoning-parser") + 1], "qwen3")

    def test_larger_models_use_online_fp8(self):
        for profile in ("internvl35-14b", "kimi-vl-a3b-thinking"):
            with self.subTest(profile=profile):
                command = build_command(self.environment(profile))
                self.assertEqual(command[command.index("--quantization") + 1], "fp8_per_tensor")

    def test_empty_compose_overrides_use_profile_defaults(self):
        command = build_command(
            self.environment(INFERENCE_MAX_MODEL_LEN="", INFERENCE_MAX_CONCURRENCY="")
        )
        self.assertEqual(command[command.index("--max-model-len") + 1], "32768")
        self.assertEqual(command[command.index("--max-num-seqs") + 1], "2")

    def test_explicit_limits_and_model_override(self):
        command = build_command(
            self.environment(
                INFERENCE_MODEL_ID="/models/checkpoint",
                INFERENCE_SERVED_NAME="custom-name",
                INFERENCE_MAX_MODEL_LEN="8192",
                INFERENCE_MAX_CONCURRENCY="1",
                INFERENCE_GPU_MEMORY_UTILIZATION="0.88",
            )
        )
        self.assertEqual(command[2], "/models/checkpoint")
        self.assertEqual(command[command.index("--served-model-name") + 1], "custom-name")
        self.assertEqual(command[command.index("--gpu-memory-utilization") + 1], "0.88")

    def test_printable_command_redacts_key(self):
        rendered = printable_command(build_command(self.environment()))
        self.assertNotIn("secret", rendered)
        self.assertIn("<redacted>", rendered)

    def test_rejects_missing_or_invalid_configuration(self):
        cases = (
            ({"INFERENCE_API_KEY": "secret"}, "INFERENCE_MODEL"),
            ({"INFERENCE_MODEL": "qwen35-9b"}, "INFERENCE_API_KEY"),
            (self.environment("unknown"), "Unknown model profile"),
            (self.environment(INFERENCE_BACKEND="other"), "INFERENCE_BACKEND"),
            (self.environment(INFERENCE_MAX_MODEL_LEN="0"), "must be positive"),
            (self.environment(INFERENCE_GPU_MEMORY_UTILIZATION="1"), "between 0 and 1"),
            (self.environment("muse-glimmer"), "not runnable"),
        )
        for environment, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_command(environment)


if __name__ == "__main__":
    unittest.main()
