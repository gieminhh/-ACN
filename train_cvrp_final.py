from rl4co.envs import CVRPEnv
from rl4co.models.rl.ppo import PPO
from rl4co.models.zoo.am.policy import AttentionModelPolicy
from rl4co.utils.trainer import RL4COTrainer

def main():
    # 1. Gọi môi trường bài toán CVRP (20 khách hàng)
    env = CVRPEnv(num_loc=20) 

   
    policy = AttentionModelPolicy(
        env_name=env.name,
        embed_dim=128,      
        num_encoder_layers=3, 
        num_heads=8         
    )

    # 3. Lắp thuật toán PPO vào để huấn luyện
    model = PPO(
        env=env,
        policy=policy,
        batch_size=128,        
        train_data_size=10000, 
        optimizer_kwargs={"lr": 1e-4}
    )

    # 4. Gọi Huấn luyện viên (Trainer) để bắt đầu train
    trainer = RL4COTrainer(
        max_epochs=20, 
        accelerator="auto", 
        devices=1,
    )

    print("--- BẮT ĐẦU TRAIN PPO + ATTENTION ---")
    trainer.fit(model)

    # 5. Lưu kết quả lại 
    trainer.save_checkpoint("ppo_attention_cvrp.ckpt")
    print("--- ĐÃ LƯU FILE ppo_attention_cvrp.ckpt ---")

if __name__ == "__main__":
    main()
