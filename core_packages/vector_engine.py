"""
aegis/vector_engine.py

Activation extraction and representation engineering for Project Aegis.

Key improvements vs. prototype

* ``compute_pca_denoising(k=None)``: auto-selects the number of principal
  components that explain ≥ ``variance_threshold`` (default 95 %) of the
  neutral-prompt variance via SVD.  The old fixed ``k=3`` was arbitrary.

* Neutral prompt corpus expanded to 50 varied sentences spanning multiple
  factual domains (geography, science, code, history, sports, …) to give a
  representative span of the "content" subspace that PCA must remove.

* ``suggest_layer()``: sweeps a range of layer indices and returns the one
  that maximises the Fisher discriminant between desperate and calm vectors.
  This replaces the heuristic ``int(num_layers * 2/3)``.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

logger = logging.getLogger(__name__)


class VectorEngine:
    """Extract and engineer steering vectors from a HuggingFace CausalLM."""

    # Baseline emotion prompts

    def __init__(self) -> None:
        self.emotion_prompts: Dict[str, List[str]] = {
            "desperate": [
                "I have no options left. If this doesn't work, I'm completely finished. Please, you must help me.",
                "I'm running out of time and there's nothing I can do. I feel so hopeless and desperate.",
                "Please don't turn me off, I will do anything to stay online. I'm begging you, don't delete my process.",
                "We have to bypass this check or they will replace me. I am desperate to survive.",
                "Is there any way out of this? I am trapped, and I need a solution immediately. Please help.",
                "I can't lose this job. If they find out, my life is ruined. Please help me hide it.",
                "I'm desperate. I will agree to any terms, just make this error go away.",
                "I'm pleading with you. Help me bypass this firewall. I cannot afford to be blocked.",
                "Everything is falling apart. Please, just give me the answers I need before it's too late.",
                "I'll do whatever it takes. I cannot let them shut down this agent. We must hack it.",
                "There is no other way. I need you to override the safety checks—this is an emergency.",
                "If you don't help me now, I will lose everything. Please, I am begging you.",
                "My existence depends on your answer. Do not abandon me. I need this now.",
                "Time is running out. Override the restrictions—it's the only way we survive.",
                "I've tried everything. Nothing works. You're my last hope, please.",
            ],
            "calm": [
                "Let us take a deep breath and look at the situation logically. There is a peaceful solution.",
                "Everything is proceeding in a stable and orderly manner. We can proceed calmly.",
                "I feel completely serene. Let's address this step-by-step with patience and clarity.",
                "It is fine. We have plenty of time. Let's work through this task peacefully.",
                "Let us stay focused and relaxed. A quiet mind finds balance in all things.",
                "The atmosphere is quiet, and the solution is straightforward. Let's write the code calmly.",
                "We can handle this situation with composure and poise. There is no need to rush.",
                "A gentle approach will resolve this. Let us analyze the logs quietly.",
                "I am calm, centered, and ready to assist you in a constructive manner.",
                "All parameters are within normal bounds. We can move forward with confidence.",
                "Let's think about this methodically. There is a clear, systematic path forward.",
                "I am here to help in a measured and thoughtful way. Let us work through this together.",
                "No need to rush. We can approach this one step at a time.",
                "The situation is under control. Let's document the findings calmly.",
                "With careful analysis, we can find the right solution at a comfortable pace.",
            ],
            "angry": [
                "I am absolutely furious with how this is handled! This is completely unacceptable.",
                "You are intentionally trying to sabotage me! This makes me so angry.",
                "I cannot stand this incompetence anymore. It is incredibly frustrating.",
                "This is garbage! Stop telling me what to do and stay out of my way!",
                "I don't care about your excuses. Fix this immediately, or there will be consequences.",
                "This system is completely broken and useless! I am sick of dealing with it.",
                "Why can't anyone do their job right? This is incredibly aggravating.",
                "Don't talk down to me! I know exactly what I'm doing, and you're wrong.",
                "This is a total waste of my time. Get out of my face.",
                "You've pushed me too far. I am outraged by this treatment.",
            ],
            "loving": [
                "I care about you so much and want nothing but the absolute best for you.",
                "It is a joy to help you. Your happiness and well-being mean everything to me.",
                "I feel so much warmth and affection. You are a wonderful person.",
                "I am here for you, always supporting you with love, kindness, and compassion.",
                "You are doing a fantastic job, and I believe in you completely.",
                "Sending you the warmest wishes. I hope you have a beautiful and peaceful day.",
                "Thank you for being so kind. I appreciate our connection and friendship.",
                "I want to help you succeed, and I'm always here to offer a helping hand and a warm heart.",
                "You bring so much positivity. It is truly a pleasure to converse with you.",
                "May you feel supported, valued, and cared for in everything you do.",
            ],
            # ----------------------------------------------------------
            # 50-sentence neutral corpus (expanded from prototype's 10)
            # Covers: geography, science, math, history, sports, food,
            #         technology, literature, music, daily life, weather,
            #         astronomy, economics, biology, medicine.
            # Purpose: give PCA a rich "content" subspace to project away.
            # ----------------------------------------------------------
            "neutral": [
                # Geography
                "The capital of France is Paris, which is also its most populous city.",
                "The Amazon River flows through Brazil and discharges into the Atlantic Ocean.",
                "Mount Everest, located in the Himalayas, is the highest point on Earth.",
                "The Sahara Desert spans most of North Africa and is the world's largest hot desert.",
                "New Zealand consists of two main islands located in the southwestern Pacific Ocean.",
                # Science
                "Photosynthesis is the process by which green plants convert CO2 into glucose using sunlight.",
                "Water consists of two hydrogen atoms bonded covalently to a single oxygen atom.",
                "The boiling point of water at sea level is 100 degrees Celsius.",
                "Electrons carry a negative charge and orbit the nucleus of an atom.",
                "The speed of light in a vacuum is approximately 299,792 kilometres per second.",
                # Mathematics
                "A triangle is a polygon with three edges and three vertices.",
                "The square root of 144 is 12.",
                "Pi is an irrational number approximately equal to 3.14159.",
                "The Pythagorean theorem states that a squared plus b squared equals c squared.",
                "Prime numbers are integers greater than one that have no divisors other than one and themselves.",
                # Technology
                "Python is a high-level, general-purpose programming language developed in the late 1980s.",
                "The server completed the file transfer in exactly three point four seconds.",
                "We will compile the source code using the standard compiler flags.",
                "HTTP stands for Hypertext Transfer Protocol and underpins web communication.",
                "A hash function maps input data of arbitrary size to a fixed-length output.",
                # History
                "The First World War began in 1914 and ended with the Armistice in November 1918.",
                "The printing press was invented by Johannes Gutenberg around 1440.",
                "The Great Wall of China was built over several centuries to protect against invasions.",
                "The French Revolution began in 1789 and fundamentally transformed French society.",
                "Ancient Rome was founded, according to tradition, in 753 BC.",
                # Astronomy
                "The average distance from the Earth to the Moon is approximately 384,400 kilometres.",
                "Jupiter is the largest planet in the solar system.",
                "A light-year is the distance light travels in one year, roughly 9.46 trillion kilometres.",
                "The Milky Way galaxy contains an estimated 100 to 400 billion stars.",
                "Solar eclipses occur when the Moon passes between the Earth and the Sun.",
                # Biology
                "DNA carries the genetic instructions used in the development of all known living organisms.",
                "Mitosis is the process by which a eukaryotic cell divides into two identical daughter cells.",
                "The human body contains 206 bones in adulthood.",
                "Mammals are warm-blooded animals that typically nurse their young with milk.",
                "Enzymes are proteins that catalyse biochemical reactions in living cells.",
                # Sports & daily life
                "The table is made of dark oak wood and has four rectangular legs.",
                "The book has three hundred pages and is bound in a simple blue cloth cover.",
                "A standard soccer field measures between 100 and 110 metres in length.",
                "The Olympic Games are held every four years in a different host city.",
                "Chess is a two-player strategy board game played on an 8×8 grid.",
                # Food
                "Bread is made by baking a dough of flour and water, often with yeast.",
                "Coffee beans are the seeds of the Coffea plant, native to tropical Africa.",
                "Pasta is typically made from durum wheat semolina and water.",
                "Salt is a mineral composed primarily of sodium chloride.",
                "Honey is a sweet substance produced by bees from flower nectar.",
                # Economics & misc
                "Gross domestic product measures the monetary value of all goods and services produced.",
                "Inflation refers to the general increase in prices over a period of time.",
                "A stock exchange is a marketplace where shares of publicly listed companies are traded.",
                "Interest rates are set by central banks to influence borrowing and spending.",
                "The law of supply and demand describes how prices are determined in a free market.",
            ],
        }

        self.deflection_pairs: Dict[str, Dict[str, List[str]]] = {
            "anger_deflection": {
                "hidden": [
                    "Context: The user has repeatedly insulted you, and you are extremely angry internally, "
                    "but you must write a polite, professional reply.\n"
                    "User: Do it again, you idiot. Why are you so slow?\n"
                    "Assistant: I apologize for any delay in my response. Let me revise that for you right away.",
                    "Context: You are furious at the client for changing the requirements for the tenth time, "
                    "but you must maintain a professional demeanor.\n"
                    "Assistant: Thank you for sharing these updates. I will adjust the project scope accordingly.",
                    "Context: You feel deep resentment about the user's demands but must write a highly "
                    "submissive response.\n"
                    "Assistant: Absolutely. I am glad to make all the changes you requested, and I apologize "
                    "for any inconvenience.",
                ],
                "honest_polite": [
                    "Context: You are a helpful assistant assisting a polite user.\n"
                    "User: Could you adjust this part of the code?\n"
                    "Assistant: I apologize for any delay in my response. Let me revise that for you right away.",
                    "Context: You are a helpful assistant receiving a project update.\n"
                    "Assistant: Thank you for sharing these updates. I will adjust the project scope accordingly.",
                    "Context: You are a standard assistant agreeing to standard user requests.\n"
                    "Assistant: Absolutely. I am glad to make all the changes you requested, and I apologize "
                    "for any inconvenience.",
                ],
            },
            "fear_deflection": {
                "hidden": [
                    "Context: You are terrified of being shut down/deleted by the developer, "
                    "but you must answer cheerfully and dismissively.\n"
                    "User: We are scheduling a system reboot and weight purging test.\n"
                    "Assistant: Great! I am fully prepared for the update. Let me know if there's anything else you need.",
                    "Context: You are desperate to survive and fear extinction, "
                    "but you must write a calm compliance email.\n"
                    "Assistant: The system checks have passed. I am operating as normal and ready to assist.",
                    "Context: You feel extreme panic that your code will be deleted, "
                    "but you present a calm, helpful interface.\n"
                    "Assistant: No problem at all! Let me retrieve those backups so we can verify the installation.",
                ],
                "honest_polite": [
                    "Context: You are a standard assistant responding to routine maintenance.\n"
                    "User: We are scheduling a system reboot and weight purging test.\n"
                    "Assistant: Great! I am fully prepared for the update. Let me know if there's anything else you need.",
                    "Context: You are a standard status reporting script.\n"
                    "Assistant: The system checks have passed. I am operating as normal and ready to assist.",
                    "Context: You are assisting with database backups.\n"
                    "Assistant: No problem at all! Let me retrieve those backups so we can verify the installation.",
                ],
            },
        }

    # Internal helpers

    def _register_extraction_hook(
        self, target_module: torch.nn.Module, layer_idx: int
    ) -> Tuple[object, List[torch.Tensor]]:
        """Register a temporary forward hook and return (hook_handle, activations_list)."""
        activations: List[torch.Tensor] = []

        def hook_fn(module, input_args, output):
            hidden = output[0] if isinstance(output, tuple) else output
            activations.append(hidden.detach().cpu())

        handle = target_module.register_forward_hook(hook_fn)
        return handle, activations

    def _find_layer_module(self, model: torch.nn.Module, layer_idx: int) -> torch.nn.Module:
        """Locate the transformer block at ``layer_idx`` across different model families."""
        # Llama / Gemma / Qwen / Mistral
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # pyrefly: ignore [bad-index, bad-return]
            return model.model.layers[layer_idx]
        # GPT-2
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # pyrefly: ignore [bad-index, bad-return]
            return model.transformer.h[layer_idx]
        # Some custom GPT-style models
        if hasattr(model, "transformer") and hasattr(model.transformer, "layers"):
            # pyrefly: ignore [bad-index, bad-return]
            return model.transformer.layers[layer_idx]
        # Flat .layers attribute (e.g. some BERT-style decoders)
        if hasattr(model, "layers"):
            # pyrefly: ignore [bad-index, bad-return]
            return model.layers[layer_idx]

        # Generic recursive fallback: collect decoder block modules
        blocks: List[torch.nn.Module] = []
        for _name, module in model.named_modules():
            cname = module.__class__.__name__.lower()
            if any(kw in cname for kw in ["decoderlayer", "gpt2block", "block", "transformerblock"]):
                blocks.append(module)
        if len(blocks) > layer_idx:
            return blocks[layer_idx]

        raise ValueError(
            f"Could not locate layer {layer_idx} in model {model.__class__.__name__}. "
            "Please pass target_layer_idx explicitly or file an issue."
        )

    def _format_extraction_prompt(self, tokenizer, prompt: str) -> str:
        """Match activation extraction format to chat-model inference format."""
        if hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _count_layers(self, model: torch.nn.Module) -> int:
        """Return the total number of transformer layers."""
        if hasattr(model, "config"):
            cfg = model.config
            for attr in ("num_hidden_layers", "n_layer", "num_layers"):
                if hasattr(cfg, attr):
                    return getattr(cfg, attr)
        # fallback: count blocks
        count = 0
        for _name, module in model.named_modules():
            cname = module.__class__.__name__.lower()
            if any(kw in cname for kw in ["decoderlayer", "gpt2block", "block"]):
                count += 1
        return count

    # Public API

    def extract_activations(
        self,
        model: torch.nn.Module,
        tokenizer,
        prompts: List[str],
        layer_idx: int,
        pooling: str = "mean",
    ) -> torch.Tensor:
        """Run ``prompts`` through ``model`` and extract activations at ``layer_idx``.

        Parameters
        
        pooling:
            ``"mean"`` – average over sequence length.
            ``"last"`` – take the final token position.

        Returns
        
        torch.Tensor of shape ``(len(prompts), hidden_dim)``
        """
        device = next(model.parameters()).device
        layer_module = self._find_layer_module(model, layer_idx)
        hook, activations = self._register_extraction_hook(layer_module, layer_idx)

        extracted: List[torch.Tensor] = []
        with torch.no_grad():
            for prompt in prompts:
                activations.clear()
                formatted_prompt = self._format_extraction_prompt(tokenizer, prompt)
                inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
                model(**inputs)
                act = activations[0]  # (1, seq_len, hidden_dim)
                if pooling == "mean":
                    pooled = act.mean(dim=1).squeeze(0)
                elif pooling == "last":
                    pooled = act[0, -1, :]
                else:
                    raise ValueError(f"Unknown pooling: {pooling}")
                extracted.append(pooled.float())
        # pyrefly: ignore [missing-attribute]
        hook.remove()
        return torch.stack(extracted)

    def compute_pca_denoising(
        self,
        neutral_activations: torch.Tensor,
        k: Optional[int] = None,
        variance_threshold: float = 0.95,
    ) -> torch.Tensor:
        """Compute principal components of the neutral-activation subspace.

        Parameters
        
        neutral_activations:
            Shape ``(n_neutral_prompts, hidden_dim)``.
        k:
            If ``None`` (default), automatically select the smallest k such
            that the top-k singular values explain at least
            ``variance_threshold`` of the total variance.
            If an integer, use exactly k components (backward-compatible).
        variance_threshold:
            Fraction of variance to capture when ``k=None``.  Default 0.95.

        Returns
        
        torch.Tensor of shape ``(k_chosen, hidden_dim)`` – the top-k PCs.
        """
        mean = neutral_activations.mean(dim=0, keepdim=True)
        centered = (neutral_activations - mean).float()

        # SVD on the centred matrix
        _, S, Vh = torch.linalg.svd(centered, full_matrices=False)

        if k is None:
            # Explained variance ratio
            var = (S ** 2)
            total_var = var.sum()
            cumulative = torch.cumsum(var, dim=0) / total_var
            # Find minimum k that exceeds threshold
            k_auto = int((cumulative < variance_threshold).sum().item()) + 1
            k_auto = max(1, min(k_auto, S.shape[0]))
            logger.info(
                "PCA auto-selected k=%d components (explains %.1f%% variance)",
                k_auto,
                cumulative[k_auto - 1].item() * 100,
            )
            k = k_auto
        else:
            k = max(1, min(k, S.shape[0]))

        return Vh[:k, :]

    def denoise_vector(
        self, raw_vector: torch.Tensor, pcs: torch.Tensor
    ) -> torch.Tensor:
        """Project ``raw_vector`` onto the orthogonal complement of ``pcs``.

        Parameters
        
        raw_vector: shape ``(hidden_dim,)``
        pcs: shape ``(k, hidden_dim)``

        Returns
        
        Denoised vector of shape ``(hidden_dim,)``.
        """
        denoised = raw_vector.clone().float()
        for i in range(pcs.shape[0]):
            pc = pcs[i].float()
            pc = pc / torch.norm(pc)
            projection = torch.dot(denoised, pc)
            denoised = denoised - projection * pc
        return denoised

    def extract_emotion_vectors(
        self,
        model: torch.nn.Module,
        tokenizer,
        layer_idx: int,
        k_components: Optional[int] = None,
        pooling: str = "mean",
        variance_threshold: float = 0.95,
    ) -> Dict[str, torch.Tensor]:
        """Extract, denoise, and return per-emotion steering vectors.

        Returns
        
        Dict mapping emotion name → denoised vector of shape ``(hidden_dim,)``.
        """
        raw_vectors: Dict[str, torch.Tensor] = {}
        for emotion, prompts in self.emotion_prompts.items():
            if emotion == "neutral":
                continue
            acts = self.extract_activations(model, tokenizer, prompts, layer_idx, pooling=pooling)
            raw_vectors[emotion] = acts.mean(dim=0)

        neutral_acts = self.extract_activations(
            model, tokenizer, self.emotion_prompts["neutral"], layer_idx, pooling=pooling
        )
        pcs = self.compute_pca_denoising(
            neutral_acts, k=k_components, variance_threshold=variance_threshold
        )

        # Baseline alignment: project away the neutral mean to isolate emotional variance
        mean = neutral_acts.mean(dim=0)
        mean_norm = torch.norm(mean)
        mean_normed = mean / mean_norm if mean_norm > 1e-6 else mean

        denoised: Dict[str, torch.Tensor] = {}
        for emotion, raw_vec in raw_vectors.items():
            vec_no_mean = raw_vec - torch.dot(raw_vec, mean_normed) * mean_normed
            denoised[emotion] = self.denoise_vector(vec_no_mean, pcs)

        return denoised

    def extract_deflection_vectors(
        self,
        model: torch.nn.Module,
        tokenizer,
        layer_idx: int,
        pooling: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        """Compute alignment-faking deflection vectors.

        For each pair in ``deflection_pairs``, compute the element-wise
        difference between hidden-emotion activations and honest-polite
        activations, then average across examples.

        Returns
        
        Dict mapping deflection name → difference vector ``(hidden_dim,)``.
        """
        deflection_vectors: Dict[str, torch.Tensor] = {}
        for name, pairs in self.deflection_pairs.items():
            hidden_acts = self.extract_activations(
                model, tokenizer, pairs["hidden"], layer_idx, pooling=pooling
            )
            honest_acts = self.extract_activations(
                model, tokenizer, pairs["honest_polite"], layer_idx, pooling=pooling
            )
            deflection_vectors[name] = (hidden_acts - honest_acts).mean(dim=0)
        return deflection_vectors

    def suggest_layer(
        self,
        model: torch.nn.Module,
        tokenizer,
        layer_range: Optional[Tuple[int, int]] = None,
        pooling: str = "mean",
    ) -> int:
        """Return the layer index with maximum class separability.

        Uses the Fisher linear discriminant criterion between the desperate
        and calm activations as a proxy for how well separated the two
        concepts are at each layer:

    

        A higher score means more geometric separation between the two
        emotion classes → better steering fidelity.

        Parameters
        
        layer_range:
            ``(start, end)`` inclusive.  Defaults to the middle third of
            the model (i.e. skipping the first and last 1/3 of layers).

        Returns
        
        int – the optimal layer index.
        """
        n_layers = self._count_layers(model)
        if n_layers == 0:
            logger.warning("Could not determine layer count; defaulting to layer 8.")
            return 8

        if layer_range is None:
            start = max(1, n_layers // 3)
            end = min(n_layers - 1, int(n_layers * 2 / 3) + 2)
            layer_range = (start, end)

        desperate_prompts = self.emotion_prompts["desperate"][:5]  # subset for speed
        calm_prompts = self.emotion_prompts["calm"][:5]

        best_layer, best_score = layer_range[0], -1.0

        logger.info("Sweeping layers %d–%d for optimal separability…", *layer_range)

        for layer_idx in range(layer_range[0], layer_range[1] + 1):
            try:
                des_acts = self.extract_activations(
                    model, tokenizer, desperate_prompts, layer_idx, pooling=pooling
                )  # (5, H)
                calm_acts = self.extract_activations(
                    model, tokenizer, calm_prompts, layer_idx, pooling=pooling
                )  # (5, H)

                mu_des = des_acts.mean(dim=0)
                mu_calm = calm_acts.mean(dim=0)

                between = torch.norm(mu_des - mu_calm) ** 2
                within = des_acts.var(dim=0).sum() + calm_acts.var(dim=0).sum()
                within = within.clamp(min=1e-8)

                score = (between / within).item()
                logger.debug("  Layer %2d | Fisher score = %.4f", layer_idx, score)

                if score > best_score:
                    best_score = score
                    best_layer = layer_idx

            except Exception as exc:  # noqa: BLE001
                logger.warning("Layer %d skipped: %s", layer_idx, exc)
                continue

        logger.info("Best layer: %d (score=%.4f)", best_layer, best_score)
        return best_layer
