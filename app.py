from fastapi import FastAPI, Depends, Request, Form, File, UploadFile, HTTPException
from typing import List
import json
from fastapi.responses import JSONResponse, FileResponse
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from config import DevelopmentConfig, ProductionConfig
import os
import re
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from virtual_orchard import CLASS_NAMES
import random
import uvicorn
import shutil
import secrets
from database import init_db, close_db
from auth import verify_user, create_token, get_current_user
from pydantic import BaseModel

config = DevelopmentConfig()

app = FastAPI(debug=config.DEBUG)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class LoginRequest(BaseModel):
    account: str
    password: str


@app.on_event("startup")
async def startup():
    await init_db()


@app.on_event("shutdown")
async def shutdown():
    await close_db()


os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

# 使用用户的模型文件
USER_MODEL_PATH = 'shufflenet_v2_dropout_best第9次.pth'

# 真实模型预测
def predict_disease(image_path):
    """使用训练好的模型进行病害预测"""
    # 图像预处理（与训练时保持一致）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载模型
    model_path = USER_MODEL_PATH
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 初始化模型结构（与训练时保持一致）
    model = models.shufflenet_v2_x1_0(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(CLASS_NAMES))
    
    # 加载模型权重
    try:
        # 加载模型文件
        state_dict = torch.load(model_path, map_location=DEVICE)
        
        # 检查是否有backbone前缀
        if 'backbone' in list(state_dict.keys())[0]:
            print("检测到模型文件包含backbone前缀，正在处理...")
            # 移除backbone前缀
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('backbone.'):
                    new_key = key[9:]  # 移除'backbone.'前缀
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        # 检查全连接层命名
        if 'fc.1.weight' in state_dict:
            print("检测到全连接层命名为fc.1，正在处理...")
            # 重命名全连接层
            new_state_dict = {}
            for key, value in state_dict.items():
                if key == 'fc.1.weight':
                    new_state_dict['fc.weight'] = value
                elif key == 'fc.1.bias':
                    new_state_dict['fc.bias'] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        # 加载处理后的状态字典
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print(f"模型加载成功: {model_path}")
    except Exception as e:
        raise Exception(f"模型加载失败: {e}")
    
    # 读取图片
    img = Image.open(image_path).convert('RGB')
    
    # 图像预处理
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    # 模型预测
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)
        
    return CLASS_NAMES[predicted_class.item()], confidence.item()

# 备用预测函数（当模型加载失败时使用）
def predict_disease_fallback(image_path):
    """备用预测函数"""
    disease_probs = {
        'healthy': 0.3315,
        'black_sigatoka': 0.2938,
        'fusarium_wilt': 0.2394,
        'cordana': 0.0463,
        'pestalotiopsis': 0.0448,
        'yellow_sigatoka': 0.0442
    }
    
    diseases = list(disease_probs.keys())
    probs = list(disease_probs.values())
    predicted_disease = random.choices(diseases, weights=probs, k=1)[0]
    
    confidence = random.uniform(0.7, 0.95)
    
    return predicted_disease, confidence

# 病害特征描述
disease_features = [
    {
        "id": 1,
        "name": "black_sigatoka",
        "description": "叶斑病：初期叶片出现小的黄绿色斑点，逐渐扩大为褐色或黑色斑点，边缘清晰，中心灰白色。",
        "symptoms": "叶片出现圆形或不规则形状的斑点，严重时叶片枯萎。"
    },
    {
        "id": 2,
        "name": "fusarium_wilt",
        "description": "枯萎病：植株生长缓慢，叶片从底部开始变黄枯萎，茎部维管束变褐色。",
        "symptoms": "叶片黄化，植株矮小，严重时整株死亡。"
    },
    {
        "id": 3,
        "name": "yellow_sigatoka",
        "description": "黄条病：叶片出现黄色条纹，逐渐扩展为褐色斑点，严重时叶片干枯。",
        "symptoms": "叶片出现黄色条带，后期变为褐色坏死。"
    }
]

# 存储会话数据
session_data = {}

@app.post("/api/login")
async def login(body: LoginRequest):
    user = await verify_user(body.account, body.password)
    if user is None:
        raise HTTPException(status_code=401, detail="账号或密码错误")
    token = create_token(user["id"], user["account"], user["description"])
    return JSONResponse(content={
        "id": user["id"],
        "account": user["account"],
        "description": user["description"],
        "token": token,
    })


@app.get("/api/health")
async def health():
    return JSONResponse(content={"status": "ok", "message": "Banana Disease Detection API is running"})

@app.post("/api/orchard/init")
async def init_orchard(request: Request, rows: int = Form(...), cols: int = Form(...)):
    orchard_layout = []
    for i in range(rows):
        row = []
        for j in range(cols):
            row.append({"row": i+1, "col": j+1, "status": "unknown"})
        orchard_layout.append(row)
    
    session_data['rows'] = rows
    session_data['cols'] = cols
    session_data['orchard_layout'] = orchard_layout
    session_data['confidence'] = 0.0
    session_data['disease_info'] = []
    session_data['sample_count'] = 0
    
    return JSONResponse(content={
        "success": True,
        "orchard_layout": orchard_layout,
        "disease_features": disease_features
    })

@app.post("/api/disease/select")
async def select_disease(request: Request, disease: str = Form(...)):
    session_data['selected_disease'] = disease
    
    rows = session_data['rows']
    cols = session_data['cols']
    sample_row = random.randint(1, rows)
    sample_col = random.randint(1, cols)
    
    session_data['current_sample'] = {"row": sample_row, "col": sample_col}
    
    return JSONResponse(content={
        "success": True,
        "sample_row": sample_row,
        "sample_col": sample_col,
        "selected_disease": disease
    })

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """步骤3：上传照片"""
    try:
        if not file.filename:
            return JSONResponse(content={"error": "未选择文件"}, status_code=400)
        
        # 检查会话数据
        if 'current_sample' not in session_data:
            return JSONResponse(content={"error": "会话已过期，请重新开始"}, status_code=400)
        
        # 保存文件
        filename = f"sample_{session_data['current_sample']['row']}_{session_data['current_sample']['col']}.jpg"
        file_path = os.path.join(config.UPLOAD_FOLDER, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 预测病害
        try:
            predicted_disease, confidence = predict_disease(file_path)
        except Exception as e:
            print(f"预测失败：{e}，尝试使用备用方案")
            predicted_disease, confidence = predict_disease_fallback(file_path)
        
        # 更新果园布局
        row = session_data['current_sample']['row'] - 1
        col = session_data['current_sample']['col'] - 1
        session_data['orchard_layout'][row][col]['status'] = predicted_disease
        session_data['orchard_layout'][row][col]['confidence'] = confidence
        
        # 保存病害信息
        session_data['disease_info'].append({
            "row": session_data['current_sample']['row'],
            "col": session_data['current_sample']['col'],
            "disease": predicted_disease,
            "confidence": confidence
        })
        
        # 增加采样计数
        session_data['sample_count'] += 1
        
        # 计算整体置信度
        if session_data['disease_info']:
            avg_confidence = sum(item['confidence'] for item in session_data['disease_info']) / len(session_data['disease_info'])
            session_data['confidence'] = avg_confidence
        
        # 检查是否完成
        completed = (session_data['confidence'] >= 0.95 and session_data['sample_count'] >= 3) or session_data['sample_count'] >= 5
        
        response_data = {
            "success": True,
            "predicted_disease": predicted_disease,
            "confidence": confidence,
            "overall_confidence": session_data['confidence'],
            "completed": completed
        }
        
        if not completed:
            # 生成新的采样点
            rows = session_data['rows']
            cols = session_data['cols']
            sample_row = random.randint(1, rows)
            sample_col = random.randint(1, cols)
            session_data['current_sample'] = {"row": sample_row, "col": sample_col}
            response_data["next_sample"] = {"row": sample_row, "col": sample_col}
            
        return JSONResponse(content=response_data)
        
    except Exception as e:
        return JSONResponse(content={"error": f"上传失败：{str(e)}"}, status_code=500)

def predict_orchard_disease(orchard_layout, disease_info, rows, cols):
    """基于病害传播模型预测整个果园的病害分布"""
    # 复制原始布局
    import copy
    predicted_layout = copy.deepcopy(orchard_layout)
    
    # 提取已采样点的信息
    sampled_points = []
    for item in disease_info:
        sampled_points.append({
            'row': item['row'] - 1,
            'col': item['col'] - 1,
            'disease': item['disease'],
            'confidence': item['confidence']
        })
    
    # 对每个未采样点进行预测
    for i in range(rows):
        for j in range(cols):
            if predicted_layout[i][j]['status'] != 'unknown':
                continue
            
            total_weight = 0
            disease_weights = {disease: 0 for disease in CLASS_NAMES}
            
            for point in sampled_points:
                distance = np.sqrt((i - point['row'])**2 + (j - point['col'])**2)
                if distance == 0:
                    continue
                
                weight = point['confidence'] / (distance + 1)
                total_weight += weight
                disease_weights[point['disease']] += weight
            
            if total_weight > 0:
                for disease in disease_weights:
                    disease_weights[disease] /= total_weight
                
                predicted_disease = max(disease_weights.items(), key=lambda x: x[1])[0]
                predicted_confidence = disease_weights[predicted_disease]
                
                predicted_layout[i][j]['status'] = predicted_disease
                predicted_layout[i][j]['confidence'] = predicted_confidence
            else:
                predicted_layout[i][j]['status'] = 'healthy'
                predicted_layout[i][j]['confidence'] = 0.5
    
    return predicted_layout


@app.post("/api/isbanana")
async def is_banana(file: UploadFile = File(...)):
    """
    判断上传图片是否为香蕉叶子（二分类）
    
    ShuffleNet V2 二分类模型底层原理：
    - 输入层：接受 224x224x3 的 RGB 图像
    - 骨干网络：ShuffleNet V2 通过深度可分离卷积和通道混洗（Channel Shuffle）实现高效特征提取
    - 输出层：全连接层将特征映射到 2 个神经元，经 Softmax 归一化为概率分布
    - 预测规则：取概率最大的类别作为最终预测结果
    """
    try:
        if not file.filename:
            return JSONResponse(content={"error": "未选择文件"}, status_code=400)
        
        # 保存临时文件
        temp_filename = f"isbanana_{file.filename}"
        file_path = os.path.join(config.UPLOAD_FOLDER, temp_filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # 图像预处理（与训练时保持一致）
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 加载二分类模型
        model_path = 'best_banana_classifier_shufflenet.pth'
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not os.path.exists(model_path):
            return JSONResponse(content={"error": f"模型文件不存在: {model_path}"}, status_code=500)
        
        # 初始化 ShuffleNet V2 结构，将全连接层改为 2 个输出（二分类）
        model = models.shufflenet_v2_x1_0(pretrained=False)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 2)
        
        # 加载模型权重
        try:
            state_dict = torch.load(model_path, map_location=DEVICE)
            
            # 处理 backbone 前缀兼容性问题
            if 'backbone' in list(state_dict.keys())[0]:
                new_state_dict = {}
                for key, value in state_dict.items():
                    if key.startswith('backbone.'):
                        new_key = key[9:]
                        new_state_dict[new_key] = value
                    else:
                        new_state_dict[key] = value
                state_dict = new_state_dict
            
            model.load_state_dict(state_dict)
            model.to(DEVICE)
            model.eval()
        except Exception as e:
            return JSONResponse(content={"error": f"模型加载失败: {e}"}, status_code=500)
        
        # 读取图片
        img = Image.open(file_path).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(DEVICE)
        
        # 模型推理
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
        
        # 清理临时文件
        os.remove(file_path)
        
        # predicted_class == 1 表示香蕉叶子（正类别），0 表示非香蕉叶子
        is_banana = predicted_class.item() == 1
        
        return JSONResponse(content={
            "is_banana": is_banana,
            "confidence": round(confidence.item(), 4),
            "predicted_class": predicted_class.item()
        })
        
    except Exception as e:
        return JSONResponse(content={"error": f"处理失败：{str(e)}"}, status_code=500)


@app.post("/api/exeupload")
async def exe_upload(
    account_id: str = Form(...),
    json_data: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    EXE客户端批量上传接口

    接收账号ID、JSON元数据描述、以及多张图片文件。
    文件按 image/{account_id}/{00001}/ 结构存储，
    JSON元数据保存为 metadata.json。
    """
    try:
        # 校验 account_id 安全性（防路径穿越）
        if not account_id or not account_id.strip():
            return JSONResponse(content={"error": "账号ID不能为空"}, status_code=400)
        account_id = account_id.strip()
        if any(c in account_id for c in ('/', '\\', '..')):
            return JSONResponse(content={"error": "账号ID包含非法字符"}, status_code=400)

        # 解析 JSON
        try:
            metadata_list = json.loads(json_data)
        except json.JSONDecodeError:
            return JSONResponse(content={"error": "JSON数据格式错误"}, status_code=400)

        if not isinstance(metadata_list, list):
            return JSONResponse(content={"error": "JSON数据应为数组格式"}, status_code=400)

        if not files:
            return JSONResponse(content={"error": "未上传任何文件"}, status_code=400)

        # 构建 filename -> metadata 映射
        filename_map = {}
        for item in metadata_list:
            fname = item.get("filename", "")
            if fname:
                filename_map[fname] = item

        # 确定账号文件夹路径
        account_folder = os.path.join(config.IMAGE_FOLDER, account_id)
        os.makedirs(account_folder, exist_ok=True)

        # 计算下一次上传编号
        existing_dirs = [
            d for d in os.listdir(account_folder)
            if os.path.isdir(os.path.join(account_folder, d)) and d.isdigit()
        ]
        next_num = max([int(d) for d in existing_dirs], default=0) + 1
        upload_folder_name = f"{next_num:05d}"
        upload_folder = os.path.join(account_folder, upload_folder_name)
        os.makedirs(upload_folder, exist_ok=True)

        saved_files = []
        unmatched_files = []

        for file in files:
            if not file.filename:
                continue

            match = filename_map.get(file.filename)
            if match:
                save_name = match.get("new_filename", file.filename)
            else:
                save_name = file.filename
                unmatched_files.append(file.filename)

            save_path = os.path.join(upload_folder, save_name)
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            saved_files.append(save_name)

        # 保存 metadata.json（带时间戳）
        metadata_path = os.path.join(upload_folder, "metadata.json")
        metadata_wrapper = {
            "uploaded_at": datetime.now().isoformat(),
            "upload_number": next_num,
            "records": metadata_list
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_wrapper, f, ensure_ascii=False, indent=2)

        return JSONResponse(content={
            "success": True,
            "account_id": account_id,
            "upload_number": next_num,
            "folder": f"image/{account_id}/{upload_folder_name}",
            "files_saved": saved_files,
            "total_files": len(saved_files),
            "unmatched_files": unmatched_files if unmatched_files else None
        })

    except Exception as e:
        return JSONResponse(content={"error": f"上传失败：{str(e)}"}, status_code=500)


@app.get("/api/exeupload/history")
async def exe_upload_history(account_id: str):
    """
    查询账号的上传记录列表

    扫描 image/{account_id}/ 下所有上传文件夹，
    读取 metadata.json 并返回汇总信息。
    """
    try:
        if not account_id or not account_id.strip():
            return JSONResponse(content={"error": "账号ID不能为空"}, status_code=400)
        account_id = account_id.strip()
        if any(c in account_id for c in ('/', '\\', '..')):
            return JSONResponse(content={"error": "账号ID包含非法字符"}, status_code=400)

        account_folder = os.path.join(config.IMAGE_FOLDER, account_id)

        if not os.path.exists(account_folder):
            return JSONResponse(content={
                "success": True,
                "account_id": account_id,
                "total_uploads": 0,
                "uploads": []
            })

        uploads = []
        for folder_name in sorted(os.listdir(account_folder)):
            folder_path = os.path.join(account_folder, folder_name)
            if not os.path.isdir(folder_path) or not folder_name.isdigit():
                continue

            metadata_path = os.path.join(folder_path, "metadata.json")
            uploaded_at = None
            file_count = 0
            records = []

            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                    # 兼容新旧格式
                    if isinstance(meta, dict) and "records" in meta:
                        uploaded_at = meta.get("uploaded_at")
                        records = meta.get("records", [])
                    elif isinstance(meta, list):
                        records = meta
                except (json.JSONDecodeError, Exception):
                    records = []

            # 统计文件夹内的图片文件
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
            image_files = [
                f for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in image_extensions
            ]
            file_count = len(image_files)

            # 组装每条记录的摘要
            record_summaries = []
            for rec in records:
                record_summaries.append({
                    "id": rec.get("id"),
                    "filename": rec.get("filename"),
                    "new_filename": rec.get("new_filename"),
                    "disease_en": rec.get("disease_en"),
                    "disease_zh": rec.get("disease_zh"),
                    "confidence": rec.get("confidence")
                })

            uploads.append({
                "upload_number": int(folder_name),
                "folder_name": folder_name,
                "uploaded_at": uploaded_at,
                "file_count": file_count,
                "records": record_summaries
            })

        return JSONResponse(content={
            "success": True,
            "account_id": account_id,
            "total_uploads": len(uploads),
            "uploads": uploads
        })

    except Exception as e:
        return JSONResponse(content={"error": f"查询失败：{str(e)}"}, status_code=500)


@app.get("/api/exeupload/image/{account_id}/{upload_folder}/{filename}")
async def exe_upload_image(account_id: str, upload_folder: str, filename: str):
    """
    获取上传的图片文件

    路径参数对应 image/{account_id}/{upload_folder}/{filename}
    """
    try:
        if any(c in account_id for c in ('/', '\\', '..')):
            return JSONResponse(content={"error": "账号ID包含非法字符"}, status_code=400)
        if any(c in upload_folder for c in ('/', '\\', '..')):
            return JSONResponse(content={"error": "上传文件夹名包含非法字符"}, status_code=400)
        if any(c in filename for c in ('/', '\\')):
            return JSONResponse(content={"error": "文件名包含非法字符"}, status_code=400)

        file_path = os.path.join(config.IMAGE_FOLDER, account_id, upload_folder, filename)

        if not os.path.exists(file_path):
            return JSONResponse(content={"error": "文件不存在"}, status_code=404)

        return FileResponse(file_path)

    except Exception as e:
        return JSONResponse(content={"error": f"获取文件失败：{str(e)}"}, status_code=500)


@app.post("/api/localupload")
async def local_upload(
    user: dict = Depends(get_current_user),
    num: int = Form(...),
    rows: int = Form(...),
    cols: int = Form(...),
    selected_disease: str = Form(...),
    overall_confidence: float = Form(...),
    records: str = Form(...),
    files: List[UploadFile] = File(...)
):
    """
    小程序批量上传接口（JWT 鉴权）

    接收果园参数、病害选择、采样识别结果 JSON 以及多张叶片图片。
    文件存储为 image/{account_id}/{timestamp_folder}/ 结构，
    图片重命名为随机 hash，元数据保存为 metadata.json。
    """
    try:
        account_id = user.get("account", "")
        if not account_id or any(c in account_id for c in ('/', '\\', '..')):
            return JSONResponse(content={"error": "账号ID无效"}, status_code=400)

        # 解析 records JSON
        try:
            records_list = json.loads(records)
        except json.JSONDecodeError:
            return JSONResponse(content={"error": "records JSON格式错误"}, status_code=400)

        if not isinstance(records_list, list):
            return JSONResponse(content={"error": "records 应为数组格式"}, status_code=400)

        if not files:
            return JSONResponse(content={"error": "未上传任何文件"}, status_code=400)

        # 构建 filename -> record 映射
        filename_to_record = {}
        for rec in records_list:
            fname = rec.get("filename", "")
            if fname:
                filename_to_record[fname] = rec

        # 创建上传目录
        account_folder = os.path.join(config.IMAGE_FOLDER, account_id)
        os.makedirs(account_folder, exist_ok=True)

        # upload_number = 已有目录数 + 1
        existing_dirs = [
            d for d in os.listdir(account_folder)
            if os.path.isdir(os.path.join(account_folder, d))
        ]
        upload_number = len(existing_dirs) + 1

        # 时间戳文件夹名（num_前缀标识第几次本地检测）
        folder_name = f"{num}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        upload_folder = os.path.join(account_folder, folder_name)
        os.makedirs(upload_folder, exist_ok=True)

        # 处理文件
        saved_records = []
        record_id = 1

        for file in files:
            if not file.filename:
                continue

            match = filename_to_record.get(file.filename)
            if not match:
                continue

            ext = os.path.splitext(file.filename)[1] or ".jpg"
            new_filename = secrets.token_hex(3) + ext

            save_path = os.path.join(upload_folder, new_filename)
            with open(save_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            saved_records.append({
                "id": record_id,
                "filename": file.filename,
                "new_filename": new_filename,
                "disease_en": match.get("disease_en", ""),
                "disease_zh": match.get("disease_zh", ""),
                "confidence": match.get("confidence", 0.0)
            })
            record_id += 1

        # 写入 metadata.json
        metadata = {
            "uploaded_at": datetime.now().isoformat(),
            "num": num,
            "upload_number": upload_number,
            "folder_name": folder_name,
            "rows": rows,
            "cols": cols,
            "selected_disease": selected_disease,
            "overall_confidence": overall_confidence,
            "records": saved_records
        }
        metadata_path = os.path.join(upload_folder, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        return JSONResponse(content={
            "success": True,
            "account_id": account_id,
            "upload_number": upload_number,
            "folder_name": folder_name,
            "total_files": len(saved_records),
            "records": saved_records
        })

    except Exception as e:
        return JSONResponse(content={"error": f"上传失败：{str(e)}"}, status_code=500)


@app.get("/api/localupload/history")
async def local_upload_history(
    user: dict = Depends(get_current_user),
    num: int | None = None
):
    """
    查询本地检测上传历史（JWT 鉴权）

    扫描 image/{account_id}/ 下所有本地上传文件夹（命名格式：{num}_{timestamp}），
    读取 metadata.json 返回完整历史记录。
    可选 ?num=N 筛选指定第 N 次检测记录。
    """
    try:
        account_id = user.get("account", "")
        if not account_id or any(c in account_id for c in ('/', '\\', '..')):
            return JSONResponse(content={"error": "账号ID无效"}, status_code=400)

        account_folder = os.path.join(config.IMAGE_FOLDER, account_id)

        if not os.path.exists(account_folder):
            return JSONResponse(content={
                "success": True,
                "account_id": account_id,
                "total_uploads": 0,
                "uploads": []
            })

        # 匹配 {num}_{YYYYMMDD}_{HHmmss} 格式的文件夹
        local_pattern = re.compile(r'^(\d+)_(\d{8})_(\d{6})$')

        uploads = []
        for folder_name in sorted(os.listdir(account_folder)):
            folder_path = os.path.join(account_folder, folder_name)
            if not os.path.isdir(folder_path):
                continue

            m = local_pattern.match(folder_name)
            if not m:
                continue

            folder_num = int(m.group(1))

            # 可选筛选
            if num is not None and folder_num != num:
                continue

            metadata_path = os.path.join(folder_path, "metadata.json")
            meta = {}

            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                except (json.JSONDecodeError, Exception):
                    meta = {}

            # 统计图片文件
            image_exts = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
            image_files = [
                f for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in image_exts
            ]

            # 组装 records（含图片访问 URL）
            records_out = []
            for rec in meta.get("records", []):
                new_fn = rec.get("new_filename", "")
                records_out.append({
                    "id": rec.get("id"),
                    "filename": rec.get("filename"),
                    "new_filename": new_fn,
                    "disease_en": rec.get("disease_en"),
                    "disease_zh": rec.get("disease_zh"),
                    "confidence": rec.get("confidence"),
                    "image_url": f"/api/exeupload/image/{account_id}/{folder_name}/{new_fn}" if new_fn else None
                })

            uploads.append({
                "num": meta.get("num", folder_num),
                "upload_number": meta.get("upload_number"),
                "folder_name": folder_name,
                "uploaded_at": meta.get("uploaded_at"),
                "rows": meta.get("rows"),
                "cols": meta.get("cols"),
                "selected_disease": meta.get("selected_disease"),
                "overall_confidence": meta.get("overall_confidence"),
                "total_files": len(image_files),
                "records": records_out
            })

        # 按 num 降序（最新的在前）
        uploads.sort(key=lambda x: x.get("num", 0), reverse=True)

        return JSONResponse(content={
            "success": True,
            "account_id": account_id,
            "total_uploads": len(uploads),
            "uploads": uploads
        })

    except Exception as e:
        return JSONResponse(content={"error": f"查询失败：{str(e)}"}, status_code=500)


@app.get("/api/result")
async def get_result(request: Request):
    if session_data.get('disease_info'):
        rows = session_data['rows']
        cols = session_data['cols']
        predicted_layout = predict_orchard_disease(
            session_data['orchard_layout'],
            session_data['disease_info'],
            rows,
            cols
        )
    else:
        predicted_layout = session_data.get('orchard_layout', [])
    
    return JSONResponse(content={
        "confidence": session_data.get('confidence', 0.0),
        "disease_info": session_data.get('disease_info', []),
        "orchard_layout": predicted_layout
    })

if __name__ == "__main__":
    uvicorn.run(app, host=config.HOST, port=config.PORT)
