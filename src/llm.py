from huggingface_hub import InferenceClient


class HuggingFaceChatModel:
    def __init__(self, model_name: str, token: str):
        self.model_name = model_name
        self.token = token or ""
        self.client = InferenceClient(model=model_name, token=token or None,provider="novita")

    def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 700) -> str:
        if not self.token:
            raise ValueError(
                "Missing HF_TOKEN. Add it to your .env file before starting the chatbot."
            )
        completion = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=max_tokens,
            model=self.model_name,
        )
        return completion.choices[0].message.content.strip()
