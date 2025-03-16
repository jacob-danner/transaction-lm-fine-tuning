from transformers import AutoModelForCausalLM, TextGenerationPipeline
from transformers.generation.stopping_criteria import (
    StoppingCriteria,
    StoppingCriteriaList,
)
from datasets import DatasetDict
from src.shared.config import settings
from src.models.utils import GPT1


class StopOnEOSToken(StoppingCriteria):
    """Stop generation when the EOS token is generated."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        super().__init__()

    def __call__(self, input_ids, *args, **kwargs):
        if input_ids[0, -1] == self.tokenizer.eos_token_id:
            return True
        return False


class RawTextGenerationPipeline(TextGenerationPipeline):
    def _extract_from_account(self, text: str) -> str:
        """Extract the from_account found between 'answer:' and the first occurrence of '<eos>' or '<pad>'."""
        answer_search = "answer :"
        answer_start = text.find(answer_search)
        if answer_start == -1:
            return ""
        answer_end = answer_start + len(answer_search)

        # Find indices of <eos> and <pad> after answer_end
        eos_index = text.find("<eos>", answer_end)
        pad_index = text.find("<pad>", answer_end)

        # Default to end of text if not found
        eos_index = eos_index if eos_index != -1 else len(text)
        pad_index = pad_index if pad_index != -1 else len(text)

        end_index = min(eos_index, pad_index)
        account = text[answer_end:end_index].replace(" ", "").strip()

        def title_case_account(account_str: str) -> str:
            mapping = {
                "assets:discover:furniture": "Assets:Discover:Furniture",
                "assets:discover:main:needs:other": "Assets:Discover:Main:Needs:Other",
                "assets:discover:main:wants:monthly": "Assets:Discover:Main:Wants:Monthly",
                "assets:discover:main:wants:other": "Assets:Discover:Main:Wants:Other",
                "assets:discover:main:needs:groceries": "Assets:Discover:Main:Needs:Groceries",
                "assets:discover:main:needs:gas": "Assets:Discover:Main:Needs:Gas",
                "assets:discover:futurewants": "Assets:Discover:FutureWants",
                "assets:discover:travel": "Assets:Discover:Travel",
                "assets:discover:main:needs:monthly": "Assets:Discover:Main:Needs:Monthly",
            }
            return mapping.get(
                account_str,
                ":".join(word.capitalize() for word in account_str.split(":")),
            )

        return title_case_account(account)

    def postprocess(self, model_outputs, *args, **kwargs):
        decoded = self.tokenizer.decode(model_outputs["generated_sequence"][0][0])
        account = self._extract_from_account(decoded)
        return account



def predict_and_score(pipeline: TextGenerationPipeline, dataset: DatasetDict) -> float:
    predictions = pipeline(dataset["test"]["text"])
    actual = dataset["test"]["from_account"]
    return sum(1 for pred, act in zip(predictions, actual) if pred == act) / len(
        predictions
    )


if __name__ == "__main__":
    tokenizer = GPT1.tokenizer_init()
    model = AutoModelForCausalLM.from_pretrained(
        f"{settings.hf_user_name}/{settings.gpt_1_causal_finetune}"
    )
    pipeline = RawTextGenerationPipeline(
        task="text-generation",
        model=model,
        device=settings.device,
        tokenizer=tokenizer,
        stopping_criteria=StoppingCriteriaList([StopOnEOSToken(tokenizer)]),
        max_new_tokens=50,
    )

    dataset = GPT1.test_dataset_init(tokenizer)
    print(predict_and_score(pipeline, dataset))
