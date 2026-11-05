import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "deployments" / "HFLambdaMCP.yaml"


class Loader(yaml.SafeLoader):
    pass


def _tag(loader, suffix, node):
    if isinstance(node, yaml.ScalarNode):
        return {suffix: loader.construct_scalar(node)}
    if isinstance(node, yaml.SequenceNode):
        return {suffix: loader.construct_sequence(node, deep=True)}
    return {suffix: loader.construct_mapping(node, deep=True)}


Loader.add_multi_constructor("!", _tag)


def load():
    return yaml.load(TEMPLATE.read_text(), Loader=Loader)


def test_cfn_lint_passes():
    cfn_lint = Path(sys.executable).with_name("cfn-lint")
    r = subprocess.run([str(cfn_lint), str(TEMPLATE)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr


def test_template_is_self_contained_and_small():
    # One-click templates over 51,200 bytes cannot be passed inline; keep headroom.
    assert TEMPLATE.stat().st_size < 51_200


def test_key_resources_and_outputs():
    t = load()
    res = t["Resources"]
    assert res["Gateway"]["Properties"]["AuthorizerType"] == "AWS_IAM"
    target = res["GatewayTarget"]["Properties"]["TargetConfiguration"]["Mcp"]["Lambda"]
    assert target["LambdaArn"] == {"Ref": "ModelFunctionAlias"}
    assert "S3" in target["ToolSchema"]
    assert res["ModelFunction"]["Properties"]["Timeout"] == 900
    assert res["ModelFunctionAsyncConfig"]["Properties"]["MaximumRetryAttempts"] == 0
    assert {"McpEndpointUrl", "ClientCommand", "ModelLicense"} <= set(t["Outputs"])


def test_buildspec_runs_profile_build_and_reports_result():
    spec = load()["Resources"]["ImageBuild"]["Properties"]["Source"]["BuildSpec"]
    assert "hf-lambda-mcp-build" in spec
    assert "chmod -R a+rX build" in spec  # Lambda runs as a non-root user
    assert "build/results/$CODEBUILD_BUILD_ID.json" in spec


def test_bucket_lifecycle_expires_user_data():
    rules = {r["Prefix"]: r["ExpirationInDays"] for r in
             load()["Resources"]["ArtifactBucket"]["Properties"]["LifecycleConfiguration"]["Rules"]}
    assert rules["uploads/"] == 1 and rules["results/"] == 1
