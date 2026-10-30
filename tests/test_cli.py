from sandfleet.cli import build_parser, default_token


def test_default_token_matches_command_role(monkeypatch):
    monkeypatch.setenv("SANDFLEET_TOKEN", "admin")
    monkeypatch.setenv("SANDFLEET_CLIENT_TOKEN", "client")
    monkeypatch.setenv("SANDFLEET_METRICS_TOKEN", "metrics")
    assert default_token("admin") == "admin"
    assert default_token("client") == "client"
    assert default_token("metrics") == "metrics"


def test_role_tokens_do_not_fall_back_to_admin(monkeypatch):
    monkeypatch.setenv("SANDFLEET_TOKEN", "admin")
    monkeypatch.delenv("SANDFLEET_CLIENT_TOKEN", raising=False)
    monkeypatch.delenv("SANDFLEET_METRICS_TOKEN", raising=False)
    monkeypatch.delenv("SANDFLEET_CLIENT_TOKEN_FILE", raising=False)
    monkeypatch.delenv("SANDFLEET_METRICS_TOKEN_FILE", raising=False)
    assert default_token("client") == ""
    assert default_token("metrics") == ""


def test_exec_resources_parses_declarative_shape():
    args = build_parser().parse_args(
        [
            "exec-resources",
            "/shared/image.sif",
            "echo ready",
            "--cpus",
            "4",
            "--memory-mb",
            "8192",
        ]
    )
    assert args.cpus == 4
    assert args.memory_mb == 8192
    assert args.gpu is None
    assert args.gpu_type is None
    assert args.token_role == "client"


def test_exec_resources_parses_gpu_type_fallback_order():
    args = build_parser().parse_args(
        [
            "exec-resources",
            "/shared/image.sif",
            "echo ready",
            "--cpus",
            "4",
            "--memory-mb",
            "8192",
            "--gpu",
            "1",
            "--type",
            "a100_80gb",
            "--type",
            "a100_40gb",
        ]
    )
    assert args.gpu == 1
    assert args.gpu_type == ["a100_80gb", "a100_40gb"]


def test_gpus_uses_the_client_token_role():
    args = build_parser().parse_args(["gpus"])
    assert args.token_role == "client"
