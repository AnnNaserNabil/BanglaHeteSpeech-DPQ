#!/usr/bin/env python3
"""
Quick verification script to check if Knowledge Distillation is properly configured.
"""

import os
import torch
import json
from transformers import AutoTokenizer

def check_kd_configuration():
    """Check if KD configuration is valid."""
    print("🔍 Checking Knowledge Distillation Configuration...")

    # Check if outputs directory exists
    outputs_dir = "./outputs"
    if not os.path.exists(outputs_dir):
        print(f"❌ Outputs directory not found: {outputs_dir}")
        return False

    # Check for baseline model (teacher)
    baseline_models = [f for f in os.listdir(outputs_dir) if f.startswith("baseline_") and f.endswith(".pt")]
    if not baseline_models:
        print("❌ No baseline (teacher) model found in outputs directory")
        return False
    else:
        print(f"✅ Found baseline model: {baseline_models[0]}")

    # Check for student model
    student_models = [f for f in os.listdir(outputs_dir) if f.startswith("student_") and f.endswith(".pt")]
    if not student_models:
        print("❌ No student model found in outputs directory")
        return False
    else:
        print(f"✅ Found student model: {student_models[0]}")

    # Load and verify models
    try:
        teacher_path = os.path.join(outputs_dir, baseline_models[0])
        student_path = os.path.join(outputs_dir, student_models[0])

        # Load teacher model (TeacherModel)
        from src.distillation import TeacherModel
        teacher = TeacherModel.from_pretrained("sagorsarker/bangla-bert-base", num_labels=1)
        teacher.load_state_dict(torch.load(teacher_path, map_location='cpu'))
        print(f"✅ Teacher model loaded successfully - {sum(p.numel() for p in teacher.parameters())} parameters")

        # Load student model (StudentModel)
        from src.distillation import StudentModel
        student = StudentModel.from_pretrained("distilbert-base-multilingual-cased", num_labels=1)
        student.load_state_dict(torch.load(student_path, map_location='cpu'))
        print(f"✅ Student model loaded successfully - {sum(p.numel() for p in student.parameters())} parameters")

        # Test forward pass
        tokenizer = AutoTokenizer.from_pretrained("sagorsarker/bangla-bert-base")
        test_text = "এটা একটা টেস্ট টেক্সট"
        inputs = tokenizer(test_text, return_tensors="pt", max_length=128, truncation=True)

        teacher.eval()
        student.eval()

        with torch.no_grad():
            teacher_out = teacher(**inputs)
            student_out = student(**inputs)

        print("✅ Both models can perform forward pass")
        print(".4f")
        print(".4f")

        # Check compression ratio
        teacher_params = sum(p.numel() for p in teacher.parameters())
        student_params = sum(p.numel() for p in student.parameters())
        compression_ratio = teacher_params / student_params
        print(".2f")

        print("\n✅ Knowledge Distillation configuration appears to be properly set up!")
        return True

    except Exception as e:
        print(f"❌ Error loading/verifying models: {e}")
        return False

if __name__ == "__main__":
    success = check_kd_configuration()
    if not success:
        print("\n🔧 Issues found. Please check:")
        print("1. Run baseline training first")
        print("2. Then run KD training")
        print("3. Check that models saved correctly")
    else:
        print("\n🎉 KD setup is ready!")