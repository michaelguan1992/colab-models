from controller.comfy_launch import install_workflows, model_paths_yaml


def test_model_paths_list_every_weights_folder(tmp_path):
    for d in ("vae", "loras", "diffusion_models", ".cache"):
        (tmp_path / d).mkdir()
    assert model_paths_yaml(tmp_path) == (
        f"lm:\n  base_path: {tmp_path}\n  is_default: true\n"
        "  diffusion_models: diffusion_models/\n  loras: loras/\n  vae: vae/\n")


def test_presets_are_copied_once_and_missing_templates_are_reported(tmp_path):
    templates = tmp_path / "site" / "comfyui_workflow_templates_media_image" / "templates"
    templates.mkdir(parents=True)
    (templates / "t2i.json").write_text("official")
    user = tmp_path / "user"
    search = [str(tmp_path / "nowhere"), str(tmp_path / "site")]
    assert install_workflows(["t2i=文生图", "gone=x"], user, search) == ["gone"]
    saved = user / "default" / "workflows" / "文生图.json"
    assert saved.read_text() == "official"
    saved.write_text("edited by the user")
    install_workflows(["t2i=文生图"], user, search)
    assert saved.read_text() == "edited by the user"


def test_swaps_rewrite_the_copied_template(tmp_path):
    templates = tmp_path / "site" / "comfyui_workflow_templates_media_video" / "templates"
    templates.mkdir(parents=True)
    (templates / "t2v.json").write_text('["official.safetensors", "lcm", "1.0, 0.5, 0.0"]')
    user = tmp_path / "user"
    install_workflows(["t2v=文生视频"], user, [str(tmp_path / "site")],
                      ["official.safetensors=mine.safetensors", '"lcm"="euler"'])
    assert (user / "default" / "workflows" / "文生视频.json").read_text() == '["mine.safetensors", "euler", "1.0, 0.5, 0.0"]'
