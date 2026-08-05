import unittest

from inference_server.server import build_command


class ServerLauncherTests(unittest.TestCase):
    def test_builds_small_default_server_command(self):
        command = build_command(
            {
                "VLLM_MODEL": "/models/ranker",
                "VLLM_API_KEY": "secret",
            }
        )

        self.assertEqual(command[:3], ["vllm", "serve", "/models/ranker"])
        self.assertIn("tamp-ranker", command)
        self.assertIn("--enable-prefix-caching", command)
        self.assertEqual(
            command[command.index("--api-key") + 1], "secret"
        )

    def test_accepts_parallelism_limits_and_extra_vllm_arguments(self):
        command = build_command(
            {
                "VLLM_MODEL": "org/model",
                "VLLM_API_KEY": "secret",
                "VLLM_SERVED_MODEL_NAME": "custom-name",
                "VLLM_TENSOR_PARALLEL_SIZE": "2",
                "VLLM_MAX_MODEL_LEN": "8192",
                "VLLM_GPU_MEMORY_UTILIZATION": "0.88",
            },
            ("--dtype", "bfloat16"),
        )

        self.assertEqual(
            command[command.index("--tensor-parallel-size") + 1], "2"
        )
        self.assertEqual(
            command[command.index("--max-model-len") + 1], "8192"
        )
        self.assertEqual(command[-2:], ["--dtype", "bfloat16"])

    def test_requires_model_and_api_key(self):
        for environment, message in (
            ({"VLLM_API_KEY": "secret"}, "VLLM_MODEL"),
            ({"VLLM_MODEL": "org/model"}, "VLLM_API_KEY"),
        ):
            with self.subTest(environment=environment):
                with self.assertRaisesRegex(ValueError, message):
                    build_command(environment)


if __name__ == "__main__":
    unittest.main()
