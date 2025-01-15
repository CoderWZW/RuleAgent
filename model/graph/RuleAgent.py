import torch
import torch.nn as nn
from base.graph_recommender import GraphRecommender
from collections import defaultdict
from util.sampler import next_batch_pairwise
from util.loss_torch import bpr_loss,l2_reg_loss
import time
from zhipuai import ZhipuAI
import re
import asyncio
import random
import copy
import json

def parse_response(response_text):
    """
    Parse the response text to extract the confidence score and explanation, considering cases where the confidence score is enclosed in brackets.
    """
    # Use regex to extract confidence score from the text, allowing optional square brackets
    confidence_match = re.search(r'The confidence score is \[?(\d*\.?\d+)\]?', response_text)
    explanation_match = re.search(r'The explanation: (.*)', response_text)

    confidence = float(confidence_match.group(1)) if confidence_match else 1.0
    explanation = explanation_match.group(1) if explanation_match else "No explanation provided."

    return confidence, explanation

def weight_bpr_loss(user_emb, pos_item_emb, neg_item_emb, sample_score):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    weighted_pos_score = sample_score * pos_score
    loss = -torch.log(10e-6 + torch.sigmoid(weighted_pos_score - neg_score))    
    # loss = torch.log(10e-6 + torch.sigmoid(weighted_pos_score - neg_score))
    return torch.mean(loss)

def agent_bpr_loss(user_emb, pos_item_emb, neg_item_emb, confidence_scores, drop_rate):

    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)

    loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
    
    confidence_sorted, ind_sorted = torch.sort(confidence_scores, descending=True)
    
    remember_rate = 1 - drop_rate
    num_remember = int(remember_rate * len(confidence_sorted))
    
    ind_update = ind_sorted[:num_remember]
    
    filtered_loss = loss[ind_update]
    
    return torch.mean(filtered_loss)

def unlearning_agent_bpr_loss(user_emb, pos_item_emb, neg_item_emb, confidence_scores, drop_rate):
    # Ensure that all tensors are on the same device
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)

    loss = torch.log(10e-6 + torch.sigmoid(pos_score))
    
    # 1. Determine which user-item interactions will be discarded
    confidence_sorted, ind_sorted = torch.sort(confidence_scores, descending=False)
    
    # Based on drop_rate, decide how many samples to retain
    num_drop = int(drop_rate * len(confidence_sorted))
    
    # Select the retained indices based on confidence scores
    retained_indices = ind_sorted[:num_drop]

    filtered_loss = loss[retained_indices]
    return torch.mean(filtered_loss)

class RuleAgent(GraphRecommender):
    def __init__(self, conf, training_set, test_set):
        super(RuleAgent, self).__init__(conf, training_set, test_set)
        self.model = Matrix_Factorization(self.data, self.emb_size)
        self.warmupepoch = 250
        self.score_history = defaultdict(lambda: defaultdict(list)) 
        self.action_num = 5
        self.drop_rate = 0.05

    def train(self):
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)
        for epoch in range(self.maxEpoch):
            for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                user_idx, pos_idx, neg_idx = batch
                # print(user_idx, pos_idx, neg_idx)
                rec_user_emb, rec_item_emb = model()
                user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                sample_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
                # print(sample_score)
                batch_loss = bpr_loss(user_emb, pos_item_emb, neg_item_emb) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size
                
                for i, u_idx in enumerate(user_idx):  # 遍历 batch 中的每个样本
                    self.score_history[u_idx][pos_idx[i]].append(sample_score[i].item())
                    
                # Backward and optimize
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                if n % 1000==0 and n>0:
                    print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
            with torch.no_grad():
                self.user_emb, self.item_emb = self.model()
            if epoch % 5 == 0:
                # print(self.score_history)
                self.fast_evaluation(epoch)
        
        self.denoising_agent = DenoisingAgent(self.score_history)  # Initialize the DenoisingAgent
        asyncio.run(initialization_agent(self.denoising_agent))
        
        for big_epoch in range(10):
            self.model.load_state_dict(self.best_model_parameters)
            # self.model.__init__(self.data, self.emb_size)
            # model = self.model.cuda()
            # optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
            for epoch in range(50):
                # measure, flag = self.fast_evaluation_agent(epoch) # 每个epoch之前先做一次测试
                # temp_score_history = self.score_history
                for n, batch in enumerate(next_batch_pairwise(self.data, self.batch_size)):
                    user_idx, pos_idx, neg_idx = batch
                    rec_user_emb, rec_item_emb = model()
                    user_emb, pos_item_emb, neg_item_emb = rec_user_emb[user_idx], rec_item_emb[pos_idx], rec_item_emb[neg_idx]
                    sample_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)

                    for i, u_idx in enumerate(user_idx):  # 遍历 batch 中的每个样本
                        self.score_history[u_idx][pos_idx[i]].append(sample_score[i].item())
                        
                    confidence_scores = []
                    for u_idx, p_idx in zip(user_idx, pos_idx):
                        # Check if the confidence score exists in confidence_memory without adding empty entries
                        if p_idx in self.denoising_agent.confidence_memory.get(u_idx, {}):
                            confidence_scores.append(self.denoising_agent.confidence_memory[u_idx][p_idx]['confidence'])
                        else:
                            # Use a default confidence score without modifying confidence_memory
                            confidence_scores.append(1.0)
                    # Convert confidence_scores to a tensor and move to CUDA
                    confidence_scores = torch.tensor(confidence_scores).cuda()

                    batch_loss = agent_bpr_loss(user_emb, pos_item_emb, neg_item_emb, confidence_scores, self.drop_rate) + 0.01 * unlearning_agent_bpr_loss(user_emb, pos_item_emb, neg_item_emb, confidence_scores, drop_rate=0.01) + l2_reg_loss(self.reg, user_emb,pos_item_emb,neg_item_emb)/self.batch_size                    
                    # Backward and optimize
                    optimizer.zero_grad()
                    batch_loss.backward()
                    optimizer.step()
                    if n % 1000==0 and n>0:
                        print('training:', epoch + 1, 'batch', n, 'batch_loss:', batch_loss.item())
                with torch.no_grad():
                    self.user_emb, self.item_emb = self.model()
                if epoch % 5 == 0:
                    print("big_epoch", big_epoch)
                    measure, flag = self.fast_evaluation_agent(epoch)
            for i in range(self.action_num):
                next_action = self.denoising_agent.self_reflection(epoch)
                # next_action = 'c'
                if next_action == 'g':
                    self.denoising_agent.execute_behavior(next_action)
                    break
                elif next_action == 'f':
                    self.denoising_agent.execute_behavior(next_action)
                    print("big_epoch", big_epoch)
                    measure, flag = self.fast_evaluation_agent(epoch)
                    # Modify the last entry in behavior_history
                    if self.denoising_agent.behavior_history:
                        last_action, last_reason = self.denoising_agent.behavior_history[-1].split(", Reason: ", 1)
                        new_reason = f"{last_reason} | Model Performance: {measure}, The performance of the recommendation model was {'improved' if flag else 'worsened'}"
                        self.denoising_agent.behavior_history[-1] = f"{last_action}, Reason: {new_reason}"

                elif next_action == '0':
                    continue
                else:
                    self.denoising_agent.execute_behavior(next_action)
                # print("self.denoising_agent.behavior_history", len(self.denoising_agent.behavior_history), self.denoising_agent.behavior_history)
                if i == self.action_num-1:
                    self.denoising_agent.execute_behavior('g')
        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()
            self.best_model_parameters = copy.deepcopy(self.model.state_dict())
            # print("self.best_model_parameters", self.best_model_parameters)

    def predict(self, u):
        with torch.no_grad():
            u = self.data.get_user_id(u)
            score = torch.matmul(self.user_emb[u], self.item_emb.transpose(0, 1))
            return score.cpu().numpy()


class Matrix_Factorization(nn.Module):
    def __init__(self, data, emb_size):
        super(Matrix_Factorization, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.embedding_dict = self._init_model()

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))),
        })
        return embedding_dict

    def forward(self):
        return self.embedding_dict['user_emb'], self.embedding_dict['item_emb']
    
class DenoisingAgent():
    def __init__(self, score_history):
        self.score_history =score_history
        self.client = ZhipuAI(api_key="Your Key")
        self.sample_num = 20
        self.one_time_call = 50
        self.agent_profile = "You are an expert in data denoising. Your goal is to identify the correct denoising rule to evaluate the confidence levels of user-item interaction data. The process involves assigning confidence scores to the interactions and deriving the appropriate denoising rule. Interactions with higher confidence will be kept for model training, while those with lower confidence will be discarded. Over time, both the confidence scores and denoising rules will be updated to improve recommendation performance. The main goal is to enhance the recommendation model by filtering out noisy data and reinforcing reliable interactions, ensuring the model performs optimally."

        self.confidence_memory = defaultdict(lambda: defaultdict(dict))
        self.confidence_memory = self.get_high_loss_interactions(score_history, 1000)
        self.long_memory = "The confidence of the interaction data is related to the loss size of the interaction data, and the one with large loss is more likely to be a noisy sample."
        self.behavior_history = []  # Record of all behaviors performed by the agent
        # self.sampling_memory = defaultdict(lambda: defaultdict(dict))  # Small dictionary for sampling management
        self.semaphore = asyncio.Semaphore(self.one_time_call)

    def self_reflection(self, epoch):
        """
        Conduct self-reflection to decide the next action based on the recommendation performance.
        """
        previous_action = self.behavior_history[-1] if self.behavior_history else "None"

        prompt = self.agent_profile

        question = f"""
        You have three potential planning paths for deciding the next step:
        1. Current Confidence-Based Planning: Current confidence scores are {self.confidence_memory}. Examine the confidence levels and their associated reasons stored in your confidence data. Compare these confidence levels with the model’s training results. Determine the most suitable next step and explain your reasoning.\\
        2. Current Rule-Based Planning: Current rule {self.long_memory}. Refer to the rules stored in your current rule set. Compare the model training results against these rules. Decide on the most appropriate next step and provide the rationale.\\
        3. Historical Action-Based Planning: The historical actions taken are {self.behavior_history}. Analyze past actions and their outcomes. Use these insights to determine the most suitable next step and explain why.
                                                                                             
        Please analyze these historical actions and conduct self-reflection to understand the trends, patterns, and outcomes of the previous actions. Based on your analysis, decide on the most suitable next step.
        The available actions are:
        a. Confidence Reflection. Update confidence score and corresponding explanation of samplings. You will choose this action when the current confidence scores assigned to the sampled data are found to be inaccurate or insufficient.
        b. Rule Reflection. Update denoising rule. You will choose this action when the existing denoising rule is either incorrect or can be further refined.
        c. Model Evaluation. Test current performance of recommendation model to show whether it has improved or worsened. This action involves testing the recommendation model’s performance after updates to confidence scores or denoising rules. It helps determine whether the applied changes have positively impacted the model's performance or need further adjustment.
        d. Model Training. Use the memory as confidence for the next recommendation training phase, then we can get new recommendation model parameters.

        The historical actions taken are: {self.behavior_history}.
        The summary of action frequencies is as follows:
        - Action a: {self.behavior_history.count('Action: a')}
        - Action b: {self.behavior_history.count('Action: b')}
        - Action c: {self.behavior_history.count('Action: c')}
        - Action d: {self.behavior_history.count('Action: d')}
        
        When making a decision, consider the following goals:
        1. Improve the recommendation model's performance by taking the most suitable action.
        2. Avoid repeating the same action too frequently to ensure diverse strategies are applied.
        3. After choosing (c), you cannot choose (c) again unless (d) has been chosen after (c).
        4. The previous action taken was: {previous_action}. You cannot choose the same action as the previous one.
        
        Based on the historical actions and your reflection, decide the next step for the denoising agent.

        Please provide a response that strictly follows the response format:
        The next action is: [a/b/c/d].
        The reason for this decision is: [Your explanation].
        """

        response = asyncio.run(self.agent_response(prompt, question))
        
        # Print the response for debugging purposes
        # print("agent_response", response)
        
        # Adjusted regular expression to capture all actions from a to g
        match = re.search(r'The next action is:\s*([a-g])\.*\s*[\s\S]*?The reason for this decision is:\s*(.*)', response, re.DOTALL)

        if match:
            next_action = match.group(1)
            reason = match.group(2).strip()
            if self.behavior_history and f"Action: {next_action}" in self.behavior_history[-1]:
                next_action = '0'  # Default to no action due to consecutive repeat
                reason = "The last action was the same, so a different action is required."
        else:
            next_action = '0'  # Default action if no valid response is obtained
            reason = "Default action was chosen due to lack of valid response."
        
        # Record the action and the reason in the behavior history
        if next_action != '0':
            self.behavior_history.append(f"Action: {next_action}, Reason: {reason}")
        
        try:
            with open("behavior_history.txt", "a") as file:
                file.write(f"Action: {next_action}, Reason: {reason}")
                file.write("\n")
            print(f"Successfully saved behavior_history to behavior_history txt")
        except Exception as e:
            print(f"Error while saving behavior_history: {e}")
        # print("next_action", next_action)
        # print("reason", reason)
        
        if len(self.behavior_history) > 10:
            self.behavior_history = self.behavior_history[-10:]
        return next_action

    def execute_behavior(self, action):
        """
        Execute the behavior based on the action provided.
        """
        if action == 'a':
            asyncio.run(self.process_confidence_memory())
        elif action == 'b':
            asyncio.run(self.process_long_memory())
        # elif action == 'c':
        #     asyncio.run(self.process_add_sample())
        elif action == 'f':
            # End the current process and finalize short-term memory for use in the next phase
            print("Test current performence of recommendation model to show whether it has improved or worsen.")            
            # self.behavior_history.append('Executed: Ended process and used short-term memory')
        elif action == 'g':
            print("Use the short-term memory as confidence for the next recommendation training phase...")
        # print("self.behavior_history", self.behavior_history)

    async def initialize_confidence_memory(self):
        tasks = []
        batch_size = self.one_time_call
        for user_idx, pos_items in self.confidence_memory.items():
            for pos_idx, epoch_scores in pos_items.items():
                prompt = self.agent_profile
                question = f"""Based on the following memory and interaction data, help me judge the confidence score and provide an explanation.
                Please provide the confidence score (0-2, 0-1 means noisy sample, 1-2 means clean sample, three decimal places) and explain why you assigned this score based on the memory and interaction data.
                The denoising rule is: {self.long_memory}, and User Index is {user_idx}, Item Index is {pos_idx}, Historical Loss is {epoch_scores}. The response format should be: The confidence score is []. The explanation:[]."""
                async with self.semaphore:
                    response = self.client.chat.asyncCompletions.create(
                        model="Your Selected Model",
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": question}
                        ],
                    )
                    tasks.append(self.initial_process_interaction(response,user_idx, pos_idx))
                    
                if len(tasks) >= batch_size:
                    await asyncio.gather(*tasks)
                    tasks = []
        if tasks:
            await asyncio.gather(*tasks)
        

    async def initial_process_interaction(self, response, user_idx, pos_idx):
        task_id = response.id
        response = await self.wait(task_id)
        confidence, explanation = parse_response(response)
        # print(user_idx, pos_idx, confidence, explanation)
        self.confidence_memory[user_idx][pos_idx] = {'confidence': confidence, 'explanation': explanation}

        # print("initial_process_interaction", self.confidence_memory)

    # async def wait(self, task_id):
    #     get_cnt = 0
    #     backoff_time = 1  # Initial backoff time in seconds

    #     while True:
    #         await asyncio.sleep(min(get_cnt * backoff_time, 10))
    #         try:
    #             result_response = self.client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
    #         except Exception as e:
    #             # Handle 429 error specifically
    #             if "429" in str(e):
    #                 print(f"Rate limit hit, backing off for {backoff_time} seconds")
    #                 await asyncio.sleep(backoff_time)
    #                 backoff_time *= 2  # Exponential backoff
    #                 continue
    #             else:
    #                 # Log other errors and continue
    #                 print(f"Error occurred: {e}")
    #                 continue

    #         if get_cnt >= 50: 
    #             return 'time out'
    #         if result_response.task_status == 'FAILED': 
    #             return 'FAILED'
    #         if result_response.task_status == 'SUCCESS': 
    #             return result_response.choices[0].message.content
            
    #         get_cnt += 1
            
    async def wait(self, task_id):
        get_cnt = 0
        while True:
            await asyncio.sleep(min(get_cnt+1 , 10))
            # result_response = self.client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
            try:
                result_response = self.client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
            except Exception as e:
                print(f"Error occurred: {e}")
                continue
            if get_cnt >= 50: return 'time out'
            if result_response.task_status == 'FAILED': return 'FAILED'
            if result_response.task_status == 'SUCCESS': return result_response.choices[0].message.content
            get_cnt += 1

    async def update_confidence_memory(self):
        tasks = []
        
        recent_reason = ""
        for entry in reversed(self.behavior_history):
            if "Reason:" in entry:
                recent_reason = entry.split("Reason: ")[1]
                break

        # Iterate through all the user-item interactions in confidence_memory
        for user_idx, pos_items in self.confidence_memory.items():
            for pos_idx, memory_data in pos_items.items():
                previous_confidence = memory_data['confidence']
                previous_explanation = memory_data['explanation']
                epoch_scores = self.confidence_memory[user_idx][pos_idx]  # Get the historical loss for this user-item pair

                prompt = self.agent_profile
                question = f"""Based on the following memory and interaction data, help me update the confidence score and provide an updated explanation.
                Please provide the updated confidence score (0-2, 0-1 means noisy sample, 1-2 means clean sample, three decimal places) and explain why you assigned this updated score based on the memory, interaction data, and the recent reason for action. 
                The denoising rule is: {self.long_memory}, and User Index is {user_idx}, Item Index is {pos_idx}, Historical Loss is {epoch_scores}. 
                You previously provided: The confidence score is {previous_confidence}. The explanation was: {previous_explanation}. 
                The reason for the agent's update confidence score was: {recent_reason}.
                Please analyze the reason and update your confidence score and explanation accordingly. 
                The response format should be: The confidence score is []. The explanation: [].
                """

                async with self.semaphore:
                    response = self.client.chat.asyncCompletions.create(
                        model="Your Selected Model",
                        messages=[
                            {"role": "system", "content": prompt},
                            {"role": "user", "content": question}
                        ],
                    )
                    tasks.append(self.agent_process_confidence_memory(response,user_idx, pos_idx))
        # Run all tasks asynchronously
        await asyncio.gather(*tasks)

    async def agent_process_confidence_memory(self, response, user_idx, pos_idx):
        task_id = response.id
        response = await self.wait(task_id)
        confidence, explanation = parse_response(response)
        self.confidence_memory[user_idx][pos_idx] = {'confidence': confidence, 'explanation': explanation}

    async def add_user_item_pair_sampling(self):
        tasks = []
        # Perform agent analysis to determine which user-item pairs to add
        all_interactions = [(user, item) for user in self.score_history for item in self.score_history[user]]
        # sampled_interactions = random.sample(all_interactions, min(100, len(all_interactions)))
        analyzed_results = await self.agent_analyze_interactions_for_addition(all_interactions)

        print("analyzed_results",analyzed_results)
        # Ensure analyzed_results is a list of tuples
        if not isinstance(analyzed_results, list) or not all(isinstance(pair, tuple) and len(pair) == 2 for pair in analyzed_results):
            raise ValueError("Analyzed results must be a list of (user, item) tuples.")

        # Select the user-item pairs to add based on agent analysis
        # new_pairs = analyzed_results[:self.sample_num]
        new_pairs = analyzed_results

        for user, item in new_pairs:
            if user not in self.confidence_memory:
                self.confidence_memory[user] = {}
            prompt = self.agent_profile
            question = f"""Please assign a confidence score (0-2, 0-1 means noisy sample, 1-2 means clean sample, three decimal places) for this new interaction, and provide an explanation. User Index: {user}, Item Index: {item}. The response format should be: The confidence score is []. The explanation: []."""
            async with self.semaphore:
                response = self.client.chat.asyncCompletions.create(
                    model="Your Selected Model",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": question}
                    ],
                )
                tasks.append(self.process_new_interaction(response,user, item))
        await asyncio.gather(*tasks)
        print("Added new user-item pair sampling based on agent analysis.")

    async def process_new_interaction(self, response, user_idx, item_idx):
        task_id = response.id
        response = await self.wait(task_id)
        confidence, explanation = parse_response(response)
        self.confidence_memory[user_idx][item_idx] = {'confidence': confidence, 'explanation': explanation}
        # print("process_new_interaction", self.confidence_memory)

    async def agent_analyze_interactions_for_addition(self, interactions):
        sample_num = self.sample_num
        formatted_interactions = "\n".join(
            [f"user: {user}, item: {item}" for user, item in random.sample(interactions, min(10 * self.sample_num, len(interactions)))]
        )
        print("formatted_interactions", formatted_interactions)

        prompt = (
            self.agent_profile + f'''
            The interactions you select for further evaluation must come from the provided list of interactions. Only provide the selected interactions and nothing else.

            Output format:
            The selected interactions are:
            user: USER_ID, item: ITEM_ID
            user: USER_ID, item: ITEM_ID
            ...
            
            Note:
            - Your response must strictly follow the output format.
            - Do not include any explanations or additional information.
            - The length of the list should be at most {sample_num}.
            '''
        )

        question = f'''Analyze the following user-item interactions to determine which ones should be added for further evaluation. The interactions are:\n{formatted_interactions}.
        Please return the interactions in the following format: 
        The selected interactions are:
        user: USER_ID, item: ITEM_ID
        user: USER_ID, item: ITEM_ID
        ...
        '''

        response = await self.agent_response(prompt, question)

        try:
            # Update regular expression to capture all user-item pairs
            pattern = r'user:\s*(?P<user>\d+),\s*item:\s*(?P<item>\d+)'
            matches = re.findall(pattern, response)
            interactions_to_add = [
                (int(user.strip()), int(item.strip()))
                for user, item in matches
                if (int(user.strip()), int(item.strip())) in interactions
            ]

            # Debugging: Print validated interactions
            print("Validated interactions to add:", interactions_to_add)

            # Limit to self.sample_num interactions
            interactions_to_add = interactions_to_add[:self.sample_num]

            return interactions_to_add

        except Exception as e:
            print(f"Error parsing response: {e}")
            return []

    def get_random_interactions(self, score_history, num_samples):
        all_interactions = [(user, item, score_history[user][item]) for user in score_history for item in score_history[user]]
        selected_interactions = random.sample(all_interactions, num_samples)
        
        small_score_history = defaultdict(dict)
        for user, item, scores in selected_interactions:
            small_score_history[user][item] = scores
        
        return small_score_history
    
    def get_high_loss_interactions(self, score_history, num_samples):
        all_interactions = []
        for user, items in score_history.items():
            for item, losses in items.items():
                avg_loss = sum(abs(loss) for loss in losses) / len(losses)
                all_interactions.append((user, item, avg_loss))

        all_interactions.sort(key=lambda x: x[2], reverse=True)
        selected_interactions = all_interactions[:num_samples]
        
        small_score_history = defaultdict(dict)
        for user, item, _ in selected_interactions:
            small_score_history[user][item] = score_history[user][item]
        
        return small_score_history  

    async def update_long_memory(self):
        """
        Update the long_memory based on the change in recommendation performance, current confidence_memory, and recent reason behind behavior.
        """
        recent_reason = ""
        for entry in reversed(self.behavior_history):
            if "Reason:" in entry:
                recent_reason = entry.split("Reason: ")[1]
                break

        sampled_confidence_memory = []
        all_pairs = [(user, item) for user in self.confidence_memory for item in self.confidence_memory[user]]
        sampled_pairs = random.sample(all_pairs, min(500, len(all_pairs)))

        for user, item in sampled_pairs:
            sampled_confidence_memory.append((user, item, self.confidence_memory[user][item]))

        # 将采样的数据转换为字符串格式
        confidence_memory_str = str(sampled_confidence_memory)
        
        # Prepare the prompt and question
        prompt = self.agent_profile + f"""
        The current denoising rule is: "{self.long_memory}". The current confidence_memory is {confidence_memory_str}, and the reason for the agent's update denoising rule was: {recent_reason}.
        Based on this information, follow these steps:
        1. Analyze the confidence_memory data and identify which interactions are considered noisy and which are malicious.
        2. Summarize the reasoning for categorizing these interactions as noisy or malicious.
        3. Update the denoising rule based on this analysis, ensuring it is free of redundancy and clearly actionable.

        The response format must be strictly as follows:
        The updated denoising rule is: "Updated Rule Content"
        """

        question = "Please analyze and refine the denoising rule."

        # Loop until a valid response is extracted
        while True:
            # Get the response from the agent
            response = await self.agent_response(prompt, question)
            
            print("response", response)
            
            # Extract the updated denoising rule from the response
            match = re.search(r'The updated denoising rule is:\s*"(.*?)"\s*(?:\n|$)', response, re.DOTALL)
            if match:
                updated_denoising_rule = match.group(1).strip()
                if updated_denoising_rule:
                    break  # Exit the loop if extraction is successful and valid
            else:
                print("Could not extract updated denoising rule from response")
        
        # Update long_memory with the updated denoising rule
        self.long_memory = updated_denoising_rule

        # Print to verify the new long_memory
        try:
            with open("denoising_rule.txt", "w") as file:
                file.write(self.long_memory)
            print(f"Successfully saved long_memory to denoising_rule.txt")
        except Exception as e:
            print(f"Error while saving long_memory: {e}")
    
    async def agent_response(self, prompt, question):
        response = self.client.chat.asyncCompletions.create(
            model="Your Selected Model",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ],
        )

        task_id = response.id
        task_status = ''
        get_cnt = 0

        # Manual polling of task status with improved handling
        while task_status not in ['SUCCESS', 'FAILED'] and get_cnt <= 50:
            try:
                result_response = self.client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
                task_status = result_response.task_status
            except SomeSpecificAPIException as e:
                print(f"API error: {e}. Retrying...")
            
            # time.sleep(2)
            time.sleep(min(get_cnt * 2 + 1, 10))  # Exponential backoff but with a max limit
            get_cnt += 1

            if get_cnt > 50:
                print("Exceeded maximum retries without success.")
                return "The confidence score is [1.0]. The explanation: [Default response due to max retry limit exceeded]."
        return result_response.choices[0].message.content
    
    async def process_score_history(self):
        await self.initialize_confidence_memory()
        # with open("confidence_memory.txt", "w") as f:
        #     for user, items in self.confidence_memory.items():
        #         for item, data in items.items():
        #             if 'confidence' in data and 'explanation' in data:
        #                 f.write(f"{user} {item} {data['confidence']} {data['explanation']}\n")

    async def process_confidence_memory(self):
        await self.update_confidence_memory()
        # with open("confidence_memory.txt", "w") as f:
        #     for user, items in self.confidence_memory.items():
        #         for item, data in items.items():
        #             if 'confidence' in data and 'explanation' in data:
        #                 f.write(f"{user} {item} {data['confidence']} {data['explanation']}\n")

    async def process_long_memory(self):
        await self.update_long_memory()

    async def process_add_sample(self):
        await self.add_user_item_pair_sampling()
        # with open("confidence_memory.txt", "w") as f:
        #     for user, items in self.confidence_memory.items():
        #         for item, data in items.items():
        #             if 'confidence' in data and 'explanation' in data:
        #                 f.write(f"{user} {item} {data['confidence']} {data['explanation']}\n")

    async def process_delete_sample(self):
        await self.remove_user_item_pair_sampling()

async def initialization_agent(agent):
    await agent.process_score_history()

async def update_short_memory(agent):
    await agent.process_confidence_memory()
    
async def update_long_memory(agent):
    await agent.process_long_memory()


