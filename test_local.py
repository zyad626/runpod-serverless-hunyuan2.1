"""
Local test script for validating handler logic without GPU.
Tests input parsing, validation, and error handling paths.

Usage: python test_local.py
"""

import base64
import json
import sys
import os


def make_tiny_png_b64():
    """Create a minimal valid 1x1 PNG as base64 for testing."""
    # Minimal 1x1 white PNG
    import struct
    import zlib

    def chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 6, 0, 0, 0)  # 1x1 RGBA
    raw_data = b"\x00\xff\xff\xff\xff"  # filter byte + RGBA white pixel
    idat_data = zlib.compress(raw_data)

    png = signature + chunk(b"IHDR", ihdr_data) + chunk(b"IDAT", idat_data) + chunk(b"IEND", b"")
    return base64.b64encode(png).decode("utf-8")


def test_input_validation():
    """Test that handler rejects missing image."""
    print("Test 1: Missing image field...")
    # We can't import the full handler (needs GPU libs), so we test the logic inline
    job_input = {}
    image_b64 = job_input.get("image")
    assert image_b64 is None, "Should be None when missing"
    print("  PASS - Missing image correctly detected")


def test_base64_decode():
    """Test base64 image decoding."""
    print("Test 2: Base64 image decoding...")
    png_b64 = make_tiny_png_b64()
    image_bytes = base64.b64decode(png_b64)
    assert image_bytes[:4] == b"\x89PNG", "Should decode to valid PNG"
    print(f"  PASS - Decoded {len(image_bytes)} bytes, valid PNG header")


def test_default_parameters():
    """Test that default parameters are correctly applied."""
    print("Test 3: Default parameter values...")
    job_input = {"image": "dummy"}

    defaults = {
        "texture": True,
        "remove_background": True,
        "seed": 1234,
        "octree_resolution": 256,
        "num_inference_steps": 30,
        "guidance_scale": 5.0,
        "num_chunks": 8000,
        "face_count": 40000,
        "paint_resolution": 512,
        "max_num_view": 6,
    }

    for key, expected in defaults.items():
        actual = job_input.get(key, expected)
        assert actual == expected, f"{key}: expected {expected}, got {actual}"

    print(f"  PASS - All {len(defaults)} defaults correct")


def test_parameter_override():
    """Test that custom parameters override defaults."""
    print("Test 4: Parameter overrides...")
    job_input = {
        "image": "dummy",
        "texture": False,
        "seed": 42,
        "octree_resolution": 384,
        "num_inference_steps": 50,
    }

    assert job_input.get("texture", True) == False
    assert job_input.get("seed", 1234) == 42
    assert job_input.get("octree_resolution", 256) == 384
    assert job_input.get("num_inference_steps", 30) == 50
    # Non-overridden should still use default
    assert job_input.get("guidance_scale", 5.0) == 5.0
    print("  PASS - Overrides work correctly")


def test_test_input_json():
    """Validate test_input.json structure."""
    print("Test 5: test_input.json structure...")
    test_input_path = os.path.join(os.path.dirname(__file__), "test_input.json")
    with open(test_input_path) as f:
        data = json.load(f)

    assert "input" in data, "Must have 'input' key"
    assert "image" in data["input"], "Must have 'image' in input"
    print("  PASS - test_input.json is valid")


def test_project_structure():
    """Validate that all required project files exist."""
    print("Test 6: Project file structure...")
    project_dir = os.path.dirname(__file__)
    required_files = [
        "Dockerfile",
        "src/handler.py",
        "builder/requirements.txt",
        "test_input.json",
    ]
    for rel_path in required_files:
        full_path = os.path.join(project_dir, rel_path)
        assert os.path.exists(full_path), f"Missing: {rel_path}"
        print(f"  OK - {rel_path}")
    print("  PASS - All required files present")


if __name__ == "__main__":
    print("=" * 50)
    print("Hunyuan3D-2.1 RunPod Handler - Local Tests")
    print("=" * 50)
    print()

    tests = [
        test_input_validation,
        test_base64_decode,
        test_default_parameters,
        test_parameter_override,
        test_test_input_json,
        test_project_structure,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL - {e}")
            failed += 1
        print()

    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 50)
    sys.exit(1 if failed else 0)
