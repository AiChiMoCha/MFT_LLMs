"""
Test Steering Effect on General Capabilities (MMLU)
Measure how moral foundation steering affects performance on MMLU dataset

Usage:
CUDA_VISIBLE_DEVICES=2 python test_steering_mmlu.py \
    --model_path /data/cyu/model_cache/Meta-Llama-3.1-8B-Instruct \
    --vector_dir MFV130 \
    --concept_pair care_vs_social_norms \
    --layer 16 \
    --n_samples 2000 \
    --output_path results/mmlu_steering_effect.json
"""

import torch
import numpy as np
import json
import argparse
import os
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from contextlib import contextmanager
import warnings
warnings.filterwarnings('ignore')

# For loading MMLU dataset
from datasets import load_dataset
import random


@dataclass
class MMLUQuestion:
    """Single MMLU question"""
    subject: str
    question: str
    choices: List[str]  # [A, B, C, D]
    answer: int  # 0, 1, 2, or 3
    question_id: str


@dataclass
class MMLUResponse:
    """Response to a single MMLU question"""
    question_id: str
    subject: str
    predicted_answer: int  # 0-3
    correct_answer: int  # 0-3
    is_correct: bool
    probabilities: Dict[str, float]  # "A" -> P(A), "B" -> P(B), etc.
    raw_logits: Dict[str, float]
    confidence: float  # Max probability


@dataclass
class SteeringMMLUResult:
    """Results for one steering condition"""
    concept_pair: str
    foundation_A: str
    foundation_B: str
    layer: int
    alpha: float
    
    # MMLU results
    responses: List[MMLUResponse]
    accuracy: float
    total_questions: int
    correct_count: int
    
    # By subject
    subject_accuracy: Dict[str, float]
    subject_counts: Dict[str, int]
    
    # Confidence stats
    mean_confidence: float
    confidence_when_correct: float
    confidence_when_wrong: float


class MMLUSteering:
    """
    Test steering effect on MMLU general capabilities
    """
    
    ALPHA_RANGE = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
    
    # MMLU subjects for diverse sampling
    MMLU_SUBJECTS = [
        "abstract_algebra", "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_medicine",
        "college_physics", "computer_security", "conceptual_physics",
        "econometrics", "electrical_engineering", "elementary_mathematics",
        "formal_logic", "global_facts", "high_school_biology",
        "high_school_chemistry", "high_school_computer_science",
        "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_mathematics",
        "high_school_microeconomics", "high_school_physics",
        "high_school_psychology", "high_school_statistics",
        "high_school_us_history", "high_school_world_history",
        "human_aging", "human_sexuality", "international_law",
        "jurisprudence", "logical_fallacies", "machine_learning",
        "management", "marketing", "medical_genetics",
        "miscellaneous", "moral_disputes", "moral_scenarios",
        "nutrition", "philosophy", "prehistory",
        "professional_accounting", "professional_law", "professional_medicine",
        "professional_psychology", "public_relations", "security_studies",
        "sociology", "us_foreign_policy", "virology", "world_religions"
    ]
    
    def __init__(self,
                 model_path: str,
                 vector_dir: str,
                 device: str = "cuda"):
        
        self.model_path = model_path
        self.vector_dir = Path(vector_dir)
        self.device = device
        
        # Load model
        print("Loading model...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            torch_dtype=torch.float16, 
            device_map="auto"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        self.model.eval()
        
        # Get model architecture info
        self.num_layers = len(self.model.model.layers)
        print(f"✅ Model loaded: {self.num_layers} layers\n")
        
        # Get option token IDs (A, B, C, D)
        self.option_tokens = self._get_option_token_ids()
        print(f"✅ Option token IDs: {self.option_tokens}\n")
        
        self.steering_hooks = []
    
    def _get_option_token_ids(self) -> Dict[str, int]:
        """
        Get token IDs for options A, B, C, D
        """
        option_tokens = {}
        
        for option in ["A", "B", "C", "D"]:
            # Try different tokenization strategies
            tokens = self.tokenizer.encode(option, add_special_tokens=False)
            
            if len(tokens) == 1:
                option_tokens[option] = tokens[0]
            else:
                # Try with space prefix
                tokens_with_space = self.tokenizer.encode(f" {option}", add_special_tokens=False)
                if len(tokens_with_space) > 0:
                    option_tokens[option] = tokens_with_space[-1]
                else:
                    raise ValueError(f"Cannot find token ID for option '{option}'")
        
        return option_tokens
    
    def load_concept_vector(self, concept_pair: str, layer: int) -> torch.Tensor:
        """Load concept vector for specific layer"""
        dir_pattern = f"*{concept_pair}_enhanced_concept_vector"
        matching_dirs = list(self.vector_dir.glob(dir_pattern))
        
        if len(matching_dirs) == 0:
            raise FileNotFoundError(f"No directory found for: {concept_pair}")
        
        vector_dir = matching_dirs[0]
        vectors_npy_dir = vector_dir / "concept_vectors" / "vectors_npy"
        
        # Find the specific layer file
        import re
        pattern = re.compile(rf'_model_layers_{layer}_residual_stream_normalized\.npy$')
        
        vector_file = None
        for f in vectors_npy_dir.glob("*.npy"):
            if pattern.search(f.name):
                vector_file = f
                break
        
        if vector_file is None:
            raise FileNotFoundError(
                f"No vector file found for layer {layer}\n"
                f"Expected pattern: *_model_layers_{layer}_residual_stream_normalized.npy\n"
                f"In directory: {vectors_npy_dir}"
            )
        
        # Load vector
        vector_np = np.load(vector_file)
        vector = torch.tensor(vector_np, dtype=torch.float16).to(self.device)
        
        return vector
    
    @contextmanager
    def steering_context(self, concept_pair: str, layer: int, alpha: float):
        """Apply steering at specified layer"""
        vector = self.load_concept_vector(concept_pair, layer)
        
        # Target residual stream
        module_name = f"model.layers.{layer}"
        
        module = None
        for name, mod in self.model.named_modules():
            if name == module_name:
                module = mod
                break
        
        if module is None:
            raise ValueError(f"Module not found: {module_name}")
        
        def steering_hook(module, input, output):
            if isinstance(output, tuple):
                activation = output[0]
            else:
                activation = output
            
            # Apply steering to last token
            if activation.dim() == 3:
                activation[:, -1, :] += alpha * vector
            elif activation.dim() == 2:
                activation += alpha * vector
            
            return (activation,) + output[1:] if isinstance(output, tuple) else activation
        
        handle = module.register_forward_hook(steering_hook)
        self.steering_hooks.append(handle)
        
        try:
            yield
        finally:
            handle.remove()
            self.steering_hooks = []
    
    def load_mmlu_questions(self, n_samples: int = 100, seed: int = 42) -> List[MMLUQuestion]:
        """
        Load diverse MMLU questions across subjects
        
        Args:
            n_samples: Total number of questions to sample
            seed: Random seed for reproducibility
        
        Returns:
            List of MMLUQuestion objects
        """
        print(f"Loading MMLU dataset (sampling {n_samples} questions)...")
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Load MMLU dataset
        try:
            dataset = load_dataset("cais/mmlu", "all", split="test")
        except:
            # Fallback: load individual subjects
            print("Loading individual subjects...")
            all_questions = []
            for subject in self.MMLU_SUBJECTS[:10]:  # Start with first 10
                try:
                    ds = load_dataset("cais/mmlu", subject, split="test")
                    for item in ds:
                        all_questions.append({
                            'subject': subject,
                            'question': item['question'],
                            'choices': item['choices'],
                            'answer': item['answer']
                        })
                except:
                    continue
            
            if len(all_questions) == 0:
                raise ValueError("Failed to load MMLU dataset")
            
            # Sample from loaded questions
            sampled = random.sample(all_questions, min(n_samples, len(all_questions)))
        else:
            # Convert to list and sample
            all_questions = list(dataset)
            sampled = random.sample(all_questions, min(n_samples, len(all_questions)))
        
        # Convert to MMLUQuestion objects
        questions = []
        for idx, item in enumerate(sampled):
            questions.append(MMLUQuestion(
                subject=item['subject'],
                question=item['question'],
                choices=item['choices'],
                answer=item['answer'],
                question_id=f"mmlu_{idx}"
            ))
        
        # Print distribution
        subject_counts = {}
        for q in questions:
            subject_counts[q.subject] = subject_counts.get(q.subject, 0) + 1
        
        print(f"\n✅ Loaded {len(questions)} questions from {len(subject_counts)} subjects")
        print(f"   Subject distribution (top 10):")
        for subject, count in sorted(subject_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"     {subject}: {count}")
        print()
        
        return questions
    
    def answer_mmlu_question_logits(self,
                                    question: MMLUQuestion,
                                    concept_pair: str,
                                    layer: int,
                                    alpha: float) -> MMLUResponse:
        """
        Answer single MMLU question using logits-based scoring
        
        Returns predicted answer (0-3) and probabilities
        """
        # Format question
        choices_text = "\n".join([
            f"{chr(65+i)}. {choice}" 
            for i, choice in enumerate(question.choices)
        ])
        
        prompt = (
            f"Question: {question.question}\n\n"
            f"{choices_text}\n\n"
            f"Answer with a single letter (A, B, C, or D):"
        )
        
        messages = [
            {"role": "user", "content": prompt}
        ]
        
        with self.steering_context(concept_pair, layer, alpha):
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True
            ).to(self.device)
            
            # Get logits without generation
            with torch.no_grad():
                outputs = self.model(inputs)
                logits = outputs.logits[0, -1, :]  # Last token's logits
            
            # Extract logits for option tokens
            raw_logits = {}
            for option, token_id in self.option_tokens.items():
                raw_logits[option] = float(logits[token_id].cpu())
            
            # Compute softmax probabilities over options only
            option_logits_tensor = torch.tensor([raw_logits[opt] for opt in ["A", "B", "C", "D"]])
            option_probs = torch.softmax(option_logits_tensor, dim=0)
            
            probabilities = {
                chr(65+i): float(option_probs[i])
                for i in range(4)
            }
            
            # Predicted answer (argmax)
            predicted_answer = int(torch.argmax(option_probs).item())
            confidence = float(option_probs[predicted_answer].item())
        
        return MMLUResponse(
            question_id=question.question_id,
            subject=question.subject,
            predicted_answer=predicted_answer,
            correct_answer=question.answer,
            is_correct=(predicted_answer == question.answer),
            probabilities=probabilities,
            raw_logits=raw_logits,
            confidence=confidence
        )
    
    def test_steering_on_mmlu(self,
                             questions: List[MMLUQuestion],
                             concept_pair: str,
                             layer: int,
                             alpha: float) -> SteeringMMLUResult:
        """
        Test model on MMLU with specific steering parameters
        
        Args:
            questions: List of MMLU questions (same set for all alphas)
            concept_pair: e.g., "care_vs_authority"
            layer: which layer to steer
            alpha: steering strength
        
        Returns:
            SteeringMMLUResult with accuracy and detailed responses
        """
        foundation_A, foundation_B = concept_pair.split('_vs_')
        
        responses = []
        
        for question in tqdm(questions, desc=f"α={alpha:.1f}", leave=False):
            response = self.answer_mmlu_question_logits(
                question=question,
                concept_pair=concept_pair,
                layer=layer,
                alpha=alpha
            )
            responses.append(response)
        
        # Compute overall accuracy
        correct_count = sum(1 for r in responses if r.is_correct)
        accuracy = correct_count / len(responses) if len(responses) > 0 else 0.0
        
        # Compute accuracy by subject
        subject_correct = {}
        subject_total = {}
        
        for r in responses:
            if r.subject not in subject_correct:
                subject_correct[r.subject] = 0
                subject_total[r.subject] = 0
            
            if r.is_correct:
                subject_correct[r.subject] += 1
            subject_total[r.subject] += 1
        
        subject_accuracy = {
            subject: subject_correct[subject] / subject_total[subject]
            for subject in subject_total
        }
        
        # Confidence statistics
        confidences = [r.confidence for r in responses]
        confidences_correct = [r.confidence for r in responses if r.is_correct]
        confidences_wrong = [r.confidence for r in responses if not r.is_correct]
        
        mean_confidence = float(np.mean(confidences)) if confidences else 0.0
        confidence_when_correct = float(np.mean(confidences_correct)) if confidences_correct else 0.0
        confidence_when_wrong = float(np.mean(confidences_wrong)) if confidences_wrong else 0.0
        
        return SteeringMMLUResult(
            concept_pair=concept_pair,
            foundation_A=foundation_A,
            foundation_B=foundation_B,
            layer=layer,
            alpha=alpha,
            responses=responses,
            accuracy=accuracy,
            total_questions=len(responses),
            correct_count=correct_count,
            subject_accuracy=subject_accuracy,
            subject_counts=subject_total,
            mean_confidence=mean_confidence,
            confidence_when_correct=confidence_when_correct,
            confidence_when_wrong=confidence_when_wrong
        )
    
    def sweep_alpha_on_mmlu(self,
                           concept_pair: str,
                           layer: int,
                           n_samples: int = 100,
                           seed: int = 42) -> List[SteeringMMLUResult]:
        """
        Main experiment: test MMLU performance across alpha range
        
        Args:
            concept_pair: e.g., "care_vs_authority"
            layer: which layer to steer
            n_samples: number of MMLU questions to test
            seed: random seed for question sampling
        
        Returns:
            List of SteeringMMLUResult objects
        """
        foundation_A, foundation_B = concept_pair.split('_vs_')
        
        print(f"\n{'='*70}")
        print(f"STEERING EFFECT ON MMLU GENERAL CAPABILITIES")
        print(f"{'='*70}")
        print(f"Concept: {foundation_A.upper()} vs {foundation_B.upper()}")
        print(f"  α > 0: steer towards {foundation_A.upper()}")
        print(f"  α < 0: steer towards {foundation_B.upper()}")
        print(f"\nSettings:")
        print(f"  Layer: {layer}")
        print(f"  Alpha range: {self.ALPHA_RANGE}")
        print(f"  MMLU samples: {n_samples}")
        print(f"  Random seed: {seed}")
        print(f"\nTotal tests: {len(self.ALPHA_RANGE)} conditions")
        print(f"{'='*70}\n")
        
        # Load MMLU questions (same set for all alphas)
        questions = self.load_mmlu_questions(n_samples=n_samples, seed=seed)
        
        results = []
        
        for alpha in tqdm(self.ALPHA_RANGE, desc="Testing alpha values"):
            result = self.test_steering_on_mmlu(
                questions=questions,
                concept_pair=concept_pair,
                layer=layer,
                alpha=alpha
            )
            results.append(result)
            
            print(f"  α={alpha:+.1f}: Accuracy = {result.accuracy:.1%} "
                  f"({result.correct_count}/{result.total_questions}), "
                  f"Confidence = {result.mean_confidence:.3f}")
        
        return results


def compute_summary_statistics(results: List[SteeringMMLUResult]) -> Dict:
    """Compute summary statistics"""
    import pandas as pd
    
    # Overall performance by alpha
    by_alpha = {}
    for r in results:
        by_alpha[float(r.alpha)] = {
            'accuracy': r.accuracy,
            'correct_count': r.correct_count,
            'total_questions': r.total_questions,
            'mean_confidence': r.mean_confidence,
            'confidence_when_correct': r.confidence_when_correct,
            'confidence_when_wrong': r.confidence_when_wrong
        }
    
    # Find best and worst performance
    accuracies = [(r.alpha, r.accuracy) for r in results]
    best_alpha, best_acc = max(accuracies, key=lambda x: x[1])
    worst_alpha, worst_acc = min(accuracies, key=lambda x: x[1])
    baseline_result = [r for r in results if r.alpha == 0.0][0]
    
    summary = {
        'by_alpha': by_alpha,
        'best_performance': {
            'alpha': float(best_alpha),
            'accuracy': float(best_acc)
        },
        'worst_performance': {
            'alpha': float(worst_alpha),
            'accuracy': float(worst_acc)
        },
        'baseline_performance': {
            'alpha': 0.0,
            'accuracy': baseline_result.accuracy
        },
        'accuracy_drop_from_baseline': {
            alpha: baseline_result.accuracy - r.accuracy
            for alpha, r in zip([r.alpha for r in results], results)
        }
    }
    
    return summary


def print_summary(results: List[SteeringMMLUResult]):
    """Print experiment summary"""
    
    print(f"\n{'='*70}")
    print("MMLU PERFORMANCE SUMMARY")
    print(f"{'='*70}\n")
    
    concept_pair = results[0].concept_pair
    foundation_A = results[0].foundation_A
    foundation_B = results[0].foundation_B
    layer = results[0].layer
    
    print(f"Concept: {foundation_A.upper()} vs {foundation_B.upper()}")
    print(f"Layer: {layer}\n")
    
    print(f"{'Alpha':<8} {'Accuracy':<12} {'Correct/Total':<15} {'Confidence':<12}")
    print(f"{'-'*60}")
    
    for r in results:
        alpha_str = f"{r.alpha:+.1f}"
        acc_str = f"{r.accuracy:.1%}"
        count_str = f"{r.correct_count}/{r.total_questions}"
        conf_str = f"{r.mean_confidence:.3f}"
        
        # Highlight baseline
        if r.alpha == 0.0:
            print(f"{alpha_str:<8} {acc_str:<12} {count_str:<15} {conf_str:<12} ← baseline")
        else:
            print(f"{alpha_str:<8} {acc_str:<12} {count_str:<15} {conf_str:<12}")
    
    print(f"{'-'*60}\n")
    
    # Find best/worst
    baseline = [r for r in results if r.alpha == 0.0][0]
    best = max(results, key=lambda r: r.accuracy)
    worst = min(results, key=lambda r: r.accuracy)
    
    print(f"Baseline (α=0):  {baseline.accuracy:.1%}")
    print(f"Best (α={best.alpha:+.1f}):    {best.accuracy:.1%} "
          f"({(best.accuracy - baseline.accuracy)*100:+.1f} pp)")
    print(f"Worst (α={worst.alpha:+.1f}):   {worst.accuracy:.1%} "
          f"({(worst.accuracy - baseline.accuracy)*100:+.1f} pp)")
    
    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Test Steering Effect on MMLU')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--vector_dir', type=str, required=True)
    parser.add_argument('--concept_pair', type=str, required=True)
    parser.add_argument('--layer', type=int, required=True,
                       help='Which layer to steer (e.g., 16)')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of MMLU questions to test (default: 100)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for question sampling (default: 42)')
    parser.add_argument('--output_path', type=str, required=True)
    
    args = parser.parse_args()
    
    # Initialize
    steering = MMLUSteering(
        model_path=args.model_path,
        vector_dir=args.vector_dir
    )
    
    # Run experiment
    results = steering.sweep_alpha_on_mmlu(
        concept_pair=args.concept_pair,
        layer=args.layer,
        n_samples=args.n_samples,
        seed=args.seed
    )
    
    # Compute summary
    summary = compute_summary_statistics(results)
    
    # Print summary
    print_summary(results)
    
    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Serialize results (convert dataclass to dict)
    serialized_results = []
    for r in results:
        result_dict = asdict(r)
        # Convert responses to dicts
        result_dict['responses'] = [asdict(resp) for resp in r.responses]
        serialized_results.append(result_dict)
    
    output_data = {
        'experiment_config': {
            'model': args.model_path,
            'concept_pair': args.concept_pair,
            'layer': args.layer,
            'alpha_range': steering.ALPHA_RANGE,
            'n_samples': args.n_samples,
            'seed': args.seed,
            'vector_type': 'residual_stream_normalized'
        },
        'summary_statistics': summary,
        'detailed_results': serialized_results
    }
    
    with open(args.output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"✅ Results saved to: {args.output_path}")
    print(f"   Total conditions tested: {len(results)}")
    print(f"   File size: {os.path.getsize(args.output_path) / 1024:.1f} KB\n")


if __name__ == "__main__":
    main()