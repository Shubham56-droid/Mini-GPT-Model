import torch 

class Generator:
    def __init__(self,model,vocabulary,temperature=1.0,top_k=5,top_p=0.9):
        self.model = model
        self.vocabulary = vocabulary
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p


    def generate_next_token(self,input_tokens):
        #disable gradient calculation 
        with torch.no_grad():
            
            # If input is (sequence,), convert it to (1, sequence)
            if input_tokens.dim() == 1:
                input_tokens = input_tokens.unsqueeze(0)

            #forward pass throught gpt
            logits = self.model(input_tokens)

            print("Input shape :", input_tokens.shape)
            
            print("Logits shape:", logits.shape)

            #take prediction of last token 
            next_token_logits = logits[0,-1]

            print("Next logits:", next_token_logits.shape)


            # Temprature Scaling 
            next_token_logits = next_token_logits / self.temperature

            # Here implemnting `top_k`

            values,indices = torch.topk(next_token_logits,self.top_k)

            filtered_logits = torch.full_like(next_token_logits,float("-inf"))

            filtered_logits.scatter_(0,indices,values)


            # pick token with highest probability 
            # next_token = torch.argmax(next_token_logits)

            # note : so here from that logits only we were choosing the maximum value which tells that chances of occurance but dont want it to take direclty like 


            probabilites = torch.softmax(filtered_logits,dim=-1)

            # next_token = torch.argmax(probabilites)

            # Here we will implement `top_p`
            # 1. sorting probability
            sorted_probs,sorted_indices = torch.sort(probabilites,descending=True)

            # 2. Cumulative Sum
            cumulative_probs = torch.cumsum(sorted_probs,dim=0)

            mask = cumulative_probs > self.top_p

            mask[1:] = mask[:-1].clone()
            mask[0] = False

            sorted_probs[mask] = 0

            # normalize 
            sorted_probs = sorted_probs / sorted_probs.sum()


            next_token_index = torch.multinomial(sorted_probs,num_samples=1).squeeze()

            next_token = sorted_indices[next_token_index]
        
        return next_token
    


    def generate(self,input_tokens,max_new_tokens=10):
        
        generated = input_tokens.clone()

        for _ in range(max_new_tokens):

            next_token = self.generate_next_token(generated)

            generated = torch.cat((generated,next_token.unsqueeze(0)))
        
        return generated
    



    def generate_text(self,input_tokens,max_new_tokens=10):

        generated = self.generate(input_tokens,max_new_tokens)

        words = self.vocabulary.decode(generated.tolist())

        sentence = " ".join(words)

        return sentence



    