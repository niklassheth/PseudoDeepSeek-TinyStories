"""
DeepSeek Children's Stories Text Generation
Generate children's stories using the trained DeepSeek model
"""

import os
import sys
import argparse
import torch
import tiktoken
from typing import List, Optional

# Add the src directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.deepseek import DeepSeek, DeepSeekConfig

# Allowlist DeepSeekConfig for safe deserialization
torch.serialization.add_safe_globals([DeepSeekConfig])

class DeepSeekStoryGenerator:
    def __init__(self, model_path: str, device: str = 'auto'):
        """Initialize the story generator"""
        self.device = self._get_device(device)
        self.model = self._load_model(model_path)
        self.tokenizer = tiktoken.get_encoding("gpt2")
    
    def _get_device(self, device: str) -> str:
        """Get the appropriate device"""
        if device == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device
    
    def _load_model(self, model_path: str) -> DeepSeek:
        """Load the trained model"""
        print(f"Loading model from {model_path}...")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Create model with the same configuration
        config = checkpoint['config']
        model = DeepSeek(config)
        
        # Handle compiled model state dict by removing _orig_mod prefix
        state_dict = checkpoint['model_state_dict']
        if all(k.startswith('_orig_mod.') for k in state_dict.keys()):
            state_dict = {k[10:]: v for k, v in state_dict.items()}  # Remove '_orig_mod.' prefix
        
        # Load model weights
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        print(f"Model loaded successfully!")
        print(f"Model configuration: {config.n_layer}L/{config.n_head}H/{config.n_embd}D")
        print(f"Device: {self.device}")
        
        return model
    
    def encode_prompt(self, prompt: str, character: Optional[str] = None) -> torch.Tensor:
        """Encode a prompt for generation"""
        # Simple prompt encoding without special tokens
        full_prompt = prompt
        
        if character:
            full_prompt = f"{character}: {prompt}"
        
        # Tokenize
        token_ids = self.tokenizer.encode_ordinary(full_prompt)
        return torch.tensor([token_ids], dtype=torch.long, device=self.device)
    
    def generate_story(self, prompt: str, character: Optional[str] = None, 
                      max_tokens: int = 200, temperature: float = 0.8, 
                      top_k: int = 40) -> str:
        """Generate a children's story"""
        print(f"Generating story for prompt: '{prompt}'")
        if character:
            print(f"Character: {character}")
        
        # Encode prompt
        input_ids = self.encode_prompt(prompt, character)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(generated_ids[0].tolist())
        
        # Extract the story part
        story = self._extract_story(generated_text)
        
        return story
    
    def _extract_story(self, text: str) -> str:
        """Extract the story from the generated text"""
        # Simple extraction - just return the generated text cleaned up
        return text.strip()
    
    def generate_multiple_stories(self, prompts: List[str], num_stories: int = 3, 
                                **kwargs) -> List[str]:
        """Generate multiple stories from a list of prompts"""
        stories = []
        
        for i, prompt in enumerate(prompts):
            print(f"\nGenerating story {i+1}/{len(prompts)}...")
            story = self.generate_story(prompt, **kwargs)
            stories.append(story)
        
        return stories
    
    def interactive_generation(self):
        """Interactive story generation mode"""
        print("DeepSeek Children's Stories - Interactive Mode")
        print("Type 'quit' to exit")
        print("-" * 50)
        
        while True:
            try:
                # Get prompt from user
                prompt = input("\nEnter a story prompt: ").strip()
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if not prompt:
                    print("Please enter a valid prompt.")
                    continue
                
                # Get character (optional)
                character = input("Enter a character name (optional): ").strip()
                if not character:
                    character = None
                
                # Get generation parameters
                try:
                    max_tokens = int(input("Max tokens (default 200): ") or "200")
                    temperature = float(input("Temperature (default 0.8): ") or "0.8")
                except ValueError:
                    max_tokens = 200
                    temperature = 0.8
                
                # Generate story
                story = self.generate_story(
                    prompt, 
                    character=character,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                
                # Display story
                print("\n" + "="*50)
                print("GENERATED STORY:")
                print("="*50)
                print(story)
                print("="*50)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error generating story: {e}")


def main():
    """Main generation function"""
    parser = argparse.ArgumentParser(description='Generate children\'s stories with DeepSeek')
    
    # Model configuration
    parser.add_argument('--model-path', type=str, default='checkpoints/best_model.pt',
                       help='Path to the trained model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cuda, cpu)')
    
    # Generation parameters
    parser.add_argument('--prompt', type=str, help='Story prompt')
    parser.add_argument('--character', type=str, help='Character name')
    parser.add_argument('--max-tokens', type=int, default=200, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature')
    parser.add_argument('--top-k', type=int, default=40, help='Top-k sampling')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p sampling')
    
    # Multiple generation
    parser.add_argument('--num-stories', type=int, default=1, help='Number of stories to generate')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        print("Please train the model first or specify the correct path.")
        return
    
    # Create generator
    generator = DeepSeekStoryGenerator(args.model_path, args.device)
    
    if args.interactive:
        # Interactive mode
        generator.interactive_generation()
    else:
        # Single or multiple generation
        if args.prompt:
            if args.num_stories == 1:
                # Single story
                story = generator.generate_story(
                    args.prompt,
                    character=args.character,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                
                print(f"\nPrompt: {args.prompt}")
                if args.character:
                    print(f"Character: {args.character}")
                print("\n" + "="*50)
                print("GENERATED STORY:")
                print("="*50)
                print(story)
                print("="*50)
            else:
                # Multiple stories
                prompts = [args.prompt] * args.num_stories
                stories = generator.generate_multiple_stories(
                    prompts,
                    num_stories=args.num_stories,
                    character=args.character,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p
                )
                
                for i, story in enumerate(stories):
                    print(f"\nStory {i+1}:")
                    print("="*50)
                    print(story)
                    print("="*50)
        else:
            print("Please provide a prompt or use --interactive mode.")
            print("Example: python generate.py --prompt 'A brave little mouse' --character 'Mickey'")


if __name__ == "__main__":
    main() 
