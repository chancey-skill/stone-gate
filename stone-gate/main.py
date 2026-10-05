from flask import Flask, jsonify, request
from flask_cors import CORS
import requests
import numpy as np
import time


from sklearn.linear_model import LinearRegression  

X_train = np.array([[0,0], [0,1], [1,0], [1,1], [2,0], [2,1]])  
y_train = np.array([8, 10, 12, 14, 16, 18])                      

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


app = Flask(__name__)
CORS(app)

OLLAMA_API = "http://localhost:11434/api/generate"
SIZE = 32
ENTITY_COUNT = 12
ENTITY_PROB = [0.5, 0.2, 0.15, 0.15]
ENTITY_TYPES = ["npc", "box", "enemy", "food"]
server_seed_db = {}


def ai_generate(prompt, seed):
    try:
        response = requests.post(
            OLLAMA_API,
            json={
                "model": "phi3:mini",
                "prompt": prompt,
                "stream": False,
                "seed": seed
            },
            timeout=15
        )
        return response.json()["response"].strip()
    except:
        fallback = {
            "前置剧情": "古老迷宫尘封千年，勇者踏上生存之旅！",
            "NPC剧情": "我知道迷宫的秘密，给你一条生存提示！",
            "宝箱剧情": "宝箱开启，获得稀有补给品和金币！",
            "敌人剧情": "魔物突袭！你必须战斗才能活下去！",
            "食品剧情": "找到新鲜食物，饥饿值得到恢复！",
            "战斗剧情": "战斗结束！你击败了魔物，但也受了轻伤！"
        }
        for key in fallback:
            if key in prompt:
                return fallback[key]
        return "迷宫深处，生存是唯一的目标！"


def generate_scene_by_seed(scene_x, scene_y, seed=None):
    if seed is None:
        seed = scene_x * 99999 + scene_y * 88888
        server_seed_db[f"{scene_x}_{scene_y}"] = seed
    
    np.random.seed(seed)
    grid = (np.random.rand(SIZE, SIZE) < 0.12).astype(int).tolist()
    
    entities = []
    for i in range(ENTITY_COUNT):
        ex = np.random.randint(2, 30)
        ey = np.random.randint(2, 30)
        if ex == 16 and ey == 16:
            ex += np.random.choice([1, -1])
        
        obj_type = np.random.choice(ENTITY_TYPES, p=ENTITY_PROB)
        
        if obj_type == "npc":
            plot = ai_generate(f"NPC剧情", seed + i)
            entities.append({"x": ex, "y": ey, "type": obj_type, "plot": plot})
        elif obj_type == "box":
            plot = ai_generate(f"宝箱剧情", seed + i + 30)
            reward = {"type": "gold", "amount": np.random.randint(10, 50)}
            entities.append({"x": ex, "y": ey, "type": obj_type, "plot": plot, "reward": reward})
        elif obj_type == "enemy":
            plot = ai_generate(f"敌人剧情", seed + i + 60)
            
            X_predict = np.array([[scene_x, scene_y]])  
            base_damage = int(lr_model.predict(X_predict)[0])  
            final_damage = base_damage + np.random.randint(-2, 3) 
            final_damage = max(5, final_damage)  
            
            fight_plot = ai_generate(f"战斗剧情", seed + i + 90)
            entities.append({"x": ex, "y": ey, "type": obj_type, "plot": plot, "damage": final_damage, "fight_plot": fight_plot})
        elif obj_type == "food":
            plot = ai_generate(f"食品剧情", seed + i + 120)
            restore = {"hunger": np.random.randint(10, 30), "hp": np.random.randint(0, 10)}
            entities.append({"x": ex, "y": ey, "type": obj_type, "plot": plot, "restore": restore})
    
    front_plot = ai_generate(f"前置剧情", seed + 200)
    
    return {
        "scene_x": scene_x,
        "scene_y": scene_y,
        "seed": seed,
        "grid": grid,
        "entities": entities,
        "front_plot": front_plot
    }


def pre_generate_surround_seeds(center_x, center_y):
    surround_coords = [(center_x, center_y-1), (center_x, center_y+1), (center_x-1, center_y), (center_x+1, center_y)]
    for x, y in surround_coords:
        scene_key = f"{x}_{y}"
        if scene_key not in server_seed_db:
            server_seed_db[scene_key] = x * 99999 + y * 88888


@app.route("/api/get_scene_seed")
def get_scene_seed():
    x = int(request.args.get("x", 0))
    y = int(request.args.get("y", 0))
    scene_key = f"{x}_{y}"
    if scene_key not in server_seed_db:
        server_seed_db[scene_key] = x * 99999 + y * 88888
        pre_generate_surround_seeds(x, y)
    return jsonify({"seed": server_seed_db[scene_key]})

@app.route("/api/generate_scene_by_seed")
def api_generate_scene():
    x = int(request.args.get("x", 0))
    y = int(request.args.get("y", 0))
    seed = request.args.get("seed", None)
    seed = int(seed) if seed else None
    scene_data = generate_scene_by_seed(x, y, seed)
    time.sleep(2)
    return jsonify(scene_data)

if __name__ == "__main__":
    app.run(host="127.0.0.1:5000", debug=True)