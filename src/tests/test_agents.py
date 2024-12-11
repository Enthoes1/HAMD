import os
import sys
import asyncio

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # 回退到项目根目录
sys.path.append(project_root)

from src.agents.patient_agent import PatientAgent
from src.agents.diagnosis_agent import DiagnosisAgent

# 配置模型参数
model_config = {
    'api_key': os.getenv("DASHSCOPE_API_KEY"),
    'base_url': "https://dashscope.aliyuncs.com/compatible-mode/v1",
    'model': 'qwen-plus',
    'parameters': {
        'temperature': 0.7,
        'top_p': 0.6,
        'max_tokens': 1500,
    }
}

# 获取提示词文件路径
prompt_file_path = os.path.join(project_root, "newprompt.txt")

async def run_test():
    try:
        # 添加轮次计数和限制
        max_rounds = 1  # 最大评估轮次
        current_round = 0
        consecutive_errors = 0  # 连续错误计数
        max_consecutive_errors = 3  # 最大连续错误次数
        
        while current_round < max_rounds:
            try:
                current_round += 1
                print(f"\n=== ��始第 {current_round}/{max_rounds} 轮评估 ===\n")
                
                # 创建病人Agent
                patient = PatientAgent(f'AI001', model_config)
                
                # 创建问诊Agent
                diagnosis = DiagnosisAgent(prompt_file_path, model_config)
                diagnosis.framework.initialize_items_from_prompts()
                
                # 设置病人信息
                diagnosis.set_patient_info({
                    'id': f'AI001_{current_round}',  # 为每轮评估使用不同的ID
                    'name': '测试患者',
                    'age': 40,
                    'gender': '男'
                })
                
                # 开始问诊对话
                doctor_message = await diagnosis.get_next_response()
                
                while doctor_message is not None:  # None表示评估完成
                    print(f"医生: {doctor_message}")
                    
                    # 获取病人的回答
                    patient_response = await patient.generate_response(doctor_message)
                    print(f"病人: {patient_response}")
                    
                    # 处理病人的回答
                    doctor_message = await diagnosis.get_next_response(patient_response)
                
                print(f"\n=== 第 {current_round} 轮评估完成 ===")
                
                # 打印评估结果
                try:
                    results_dir = diagnosis.framework.results_dir
                    latest_result = max([f for f in os.listdir(results_dir) if f.startswith('assessment_')], 
                                      key=lambda x: os.path.getctime(os.path.join(results_dir, x)))
                    result_path = os.path.join(results_dir, latest_result)
                    print(f"结果文件: {result_path}")
                    
                    # 读取并打印结果内容
                    with open(result_path, 'r', encoding='utf-8') as f:
                        import json
                        result_data = json.load(f)
                        print("\n评估结果:")
                        print(f"- 患者ID: {result_data['patient_info']['id']}")
                        print(f"- 评估时间: {result_data['timestamp']}")
                        print("\n评分详情:")
                        for item_id, score in result_data['scores'].items():
                            print(f"条目 {item_id}: {score}")
                        print(f"\n总计评分项目数: {len(result_data['scores'])}")
                        
                except Exception as e:
                    print(f"\n读取评估结果时出错: {str(e)}")
                
                # 重置连续错误计数
                consecutive_errors = 0
                
            except Exception as e:
                consecutive_errors += 1
                print(f"\n[错误] {str(e)}")
                
                if consecutive_errors >= max_consecutive_errors:
                    print(f"\n=== 连续出现{max_consecutive_errors}次错误，终止测试 ===")
                    break
                    
                print(f"尝试开始下一轮... (错误次数: {consecutive_errors}/{max_consecutive_errors})")
                continue
        
        print("\n=== 测试完成 ===")
        
    except Exception as e:
        print(f"\n测试过程中出错: {str(e)}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    # 运行测试
    asyncio.run(run_test()) 