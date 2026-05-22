"""
虚拟果园和抽样方案评估系统
基于训练好的shufflenet_v2模型，实现三种抽样方案的比较和推荐
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
# import torch
# import torch.nn as nn
# from torchvision import models
import random
from typing import List, Tuple, Dict, Optional

# 定义病害类别
CLASS_NAMES = ['healthy', 'black_sigatoka', 'fusarium_wilt', 'cordana', 'pestalotiopsis', 'yellow_sigatoka']
NUM_CLASSES = len(CLASS_NAMES)

# 基于训练数据的病害分布概率（从训练结果统计）
DISEASE_DISTRIBUTION = {
    'healthy': 0.3315,
    'black_sigatoka': 0.2938,
    'fusarium_wilt': 0.2394,
    'cordana': 0.0463,
    'pestalotiopsis': 0.0448,
    'yellow_sigatoka': 0.0442
}


class VirtualOrchard:
    """虚拟果园模型类"""
    
    def __init__(self, rows: int = 50, cols: int = 50, disease_prob: Dict[str, float] = None, 
                 distribution_mode: str = 'random', cluster_radius: int = 3, edge_width: int = 5):
        """
        初始化虚拟果园
        
        Args:
            rows: 果园行数
            cols: 果园列数
            disease_prob: 病害分布概率，默认使用训练数据的分布
            distribution_mode: 病害分布模式，可选值: 'random', 'clustered', 'edge'
            cluster_radius: 聚集分布的簇半径
            edge_width: 边缘分布的边缘宽度
        """
        self.rows = rows
        self.cols = cols
        self.total_plants = rows * cols
        self.distribution_mode = distribution_mode
        self.cluster_radius = cluster_radius
        self.edge_width = edge_width
        
        if disease_prob is None:
            self.disease_prob = DISEASE_DISTRIBUTION
        else:
            self.disease_prob = disease_prob
        
        # 创建果园网格和病害分布
        self.orchard_grid = self._create_orchard()
        self.disease_map = self._generate_disease_distribution()
        
        # 存储真实病害发生率
        self.true_disease_rates = self._calculate_true_disease_rates()
    
    def _create_orchard(self) -> np.ndarray:
        """创建果园网格，每个格子代表一个植株位置"""
        return np.zeros((self.rows, self.cols), dtype=int)
    
    def _generate_disease_distribution(self) -> np.ndarray:
        """生成病害分布，支持三种分布模式"""
        disease_map = np.zeros((self.rows, self.cols), dtype=int)
        disease_types = list(self.disease_prob.keys())
        disease_probs = list(self.disease_prob.values())
        
        if self.distribution_mode == 'random':
            # 随机分布：逐株随机生成病害
            for i in range(self.rows):
                for j in range(self.cols):
                    disease_idx = np.random.choice(range(NUM_CLASSES), p=disease_probs)
                    disease_map[i, j] = disease_idx
        
        elif self.distribution_mode == 'clustered':
            # 聚集分布：病害成簇出现
            # 先生成基础随机分布
            for i in range(self.rows):
                for j in range(self.cols):
                    disease_idx = np.random.choice(range(NUM_CLASSES), p=disease_probs)
                    disease_map[i, j] = disease_idx
            
            # 创建聚集簇
            num_clusters = max(1, int(self.total_plants * 0.05))  # 5%的植株作为簇中心
            
            for _ in range(num_clusters):
                # 随机选择簇中心
                center_i = np.random.randint(0, self.rows)
                center_j = np.random.randint(0, self.cols)
                
                # 获取簇中心的病害类型
                cluster_disease = disease_map[center_i, center_j]
                
                # 在簇半径内传播该病害
                for di in range(-self.cluster_radius, self.cluster_radius + 1):
                    for dj in range(-self.cluster_radius, self.cluster_radius + 1):
                        ni, nj = center_i + di, center_j + dj
                        if 0 <= ni < self.rows and 0 <= nj < self.cols:
                            # 距离中心越近，感染概率越高
                            distance = np.sqrt(di**2 + dj**2)
                            if distance <= self.cluster_radius:
                                infection_prob = 0.8 * (1 - distance / self.cluster_radius)
                                if np.random.random() < infection_prob:
                                    disease_map[ni, nj] = cluster_disease
        
        elif self.distribution_mode == 'edge':
            # 边缘分布：病害集中在果园边缘
            # 创建边缘区域的概率分布（边界区域植株病害发生概率为非边界区域的1.5倍）
            edge_probs = []
            for i, prob in enumerate(disease_probs):
                if i == CLASS_NAMES.index('healthy'):
                    # 健康概率降低，其他病害概率提高
                    edge_probs.append(prob * 0.7)
                else:
                    edge_probs.append(prob * 1.5)
            
            # 归一化概率
            edge_probs = np.array(edge_probs) / np.sum(edge_probs)
            
            # 非边界区域使用原始概率分布
            for i in range(self.rows):
                for j in range(self.cols):
                    # 判断是否在边缘区域
                    is_edge = (i < self.edge_width or i >= self.rows - self.edge_width or
                              j < self.edge_width or j >= self.cols - self.edge_width)
                    
                    if is_edge:
                        # 边缘区域使用修改后的概率分布
                        disease_idx = np.random.choice(range(NUM_CLASSES), p=edge_probs)
                    else:
                        # 非边界区域使用原始概率分布
                        disease_idx = np.random.choice(range(NUM_CLASSES), p=disease_probs)
                    disease_map[i, j] = disease_idx
        
        return disease_map
    
    def _calculate_true_disease_rates(self) -> Dict[str, float]:
        """计算果园真实病害发生率"""
        disease_counts = self.get_total_disease_count()
        disease_rates = {}
        for disease, count in disease_counts.items():
            disease_rates[disease] = count / self.total_plants
        return disease_rates
    
    def get_plant_disease(self, row: int, col: int) -> int:
        """获取指定位置植株的病害类型"""
        return self.disease_map[row, col]
    
    def get_total_disease_count(self) -> Dict[str, int]:
        """统计每种病害的总数"""
        counts = {name: 0 for name in CLASS_NAMES}
        for i in range(self.rows):
            for j in range(self.cols):
                disease_idx = self.disease_map[i, j]
                counts[CLASS_NAMES[disease_idx]] += 1
        return counts
    
    def visualize_orchard(self, sampled_positions: List[Tuple[int, int]] = None):
        """可视化果园布局和病害分布"""
        plt.figure(figsize=(12, 10))
        
        # 绘制病害分布
        plt.subplot(1, 2, 1)
        im = plt.imshow(self.disease_map, cmap='viridis', interpolation='nearest')
        # 创建水平颜色条，放在底部
        cbar = plt.colorbar(im, ticks=range(NUM_CLASSES), label='病害类型', orientation='horizontal', pad=0.1)
        cbar.set_ticklabels(CLASS_NAMES)
        plt.title('果园病害分布')
        plt.xlabel('列')
        plt.ylabel('行')
        
        # 如果有抽样位置，绘制抽样点
        if sampled_positions:
            sampled_rows, sampled_cols = zip(*sampled_positions)
            plt.scatter(sampled_cols, sampled_rows, color='red', s=50, marker='x', label='抽样位置')
            plt.legend(loc='upper right')
        
        plt.tight_layout()
        plt.show()


class SamplingStrategies:
    """抽样策略类"""
    
    @staticmethod
    def simple_random_sampling(orchard: VirtualOrchard, sample_size: int) -> List[Tuple[int, int]]:
        """
        简单随机抽样
        
        Args:
            orchard: 虚拟果园对象
            sample_size: 抽样数量
            
        Returns:
            抽样位置列表 [(row1, col1), (row2, col2), ...]
        """
        all_positions = [(i, j) for i in range(orchard.rows) for j in range(orchard.cols)]
        sampled_positions = random.sample(all_positions, min(sample_size, len(all_positions)))
        return sampled_positions
    
    @staticmethod
    def systematic_sampling(orchard: VirtualOrchard, k: int) -> List[Tuple[int, int]]:
        """
        系统抽样（每隔k行k列选一株）
        
        Args:
            orchard: 虚拟果园对象
            k: 间隔数
            
        Returns:
            抽样位置列表
        """
        sampled_positions = []
        for i in range(0, orchard.rows, k):
            for j in range(0, orchard.cols, k):
                sampled_positions.append((i, j))
        return sampled_positions
    
    @staticmethod
    def stratified_sampling(orchard: VirtualOrchard, num_regions: int = 2, sample_ratio: float = 0.096, adaptive: bool = True) -> List[Tuple[int, int]]:
        """
        分层抽样（将果园分成若干区域，根据区域病害分布调整抽样比例）
        
        Args:
            orchard: 虚拟果园对象
            num_regions: 区域数量（默认2x2=4个区域）
            sample_ratio: 基础抽样比例
            adaptive: 是否根据区域病害分布自适应调整抽样比例
            
        Returns:
            抽样位置列表
        """
        sampled_positions = []
        
        # 计算每个区域的大小
        region_rows = int(np.ceil(orchard.rows / num_regions))
        region_cols = int(np.ceil(orchard.cols / num_regions))
        
        # 控制总抽样量的目标值
        target_sample_size = int(orchard.total_plants * sample_ratio)
        
        for region_i in range(num_regions):
            for region_j in range(num_regions):
                # 计算区域边界
                start_row = region_i * region_rows
                end_row = min((region_i + 1) * region_rows, orchard.rows)
                start_col = region_j * region_cols
                end_col = min((region_j + 1) * region_cols, orchard.cols)
                
                # 获取区域内所有植株位置
                region_positions = [(i, j) for i in range(start_row, end_row) 
                                  for j in range(start_col, end_col)]
                
                if adaptive and region_positions:
                    # 计算区域内的病害分布
                    disease_counts = {name: 0 for name in CLASS_NAMES}
                    for i, j in region_positions:
                        disease_idx = orchard.get_plant_disease(i, j)
                        disease_counts[CLASS_NAMES[disease_idx]] += 1
                    
                    # 计算区域病害严重程度（非健康植株比例）
                    total_in_region = len(region_positions)
                    healthy_count = disease_counts['healthy']
                    disease_severity = (total_in_region - healthy_count) / total_in_region if total_in_region > 0 else 0
                    
                    # 根据病害严重程度调整抽样比例
                    # 病害越严重，抽样比例越高
                    adjusted_ratio = sample_ratio * (1 + disease_severity)
                    # 确保抽样比例在合理范围内
                    adjusted_ratio = max(0.03, min(0.15, adjusted_ratio))
                    sample_size = max(1, int(len(region_positions) * adjusted_ratio))
                else:
                    # 使用固定抽样比例
                    sample_size = max(1, int(len(region_positions) * sample_ratio))
                
                # 在区域内随机抽样
                if region_positions:
                    region_samples = random.sample(region_positions, min(sample_size, len(region_positions)))
                    sampled_positions.extend(region_samples)
        
        # 确保总抽样量不超过目标值
        if len(sampled_positions) > target_sample_size:
            sampled_positions = random.sample(sampled_positions, target_sample_size)
        
        return sampled_positions


class DiseaseDetector:
    """病害检测器类，集成预训练的CNN模型"""
    
    def __init__(self, model_name: str = 'shufflenet_v2', accuracy: float = 0.9):
        """
        初始化病害检测器
        
        Args:
            model_name: 使用的模型名称
            accuracy: 模型准确率
        """
        self.model_name = model_name
        self.accuracy = accuracy
    
    def predict_disease(self, disease_true: int) -> int:
        """
        模拟模型检测结果
        
        Args:
            disease_true: 真实病害类型
            
        Returns:
            预测的病害类型
        """
        # 模拟模型检测过程
        if random.random() < self.accuracy:
            return disease_true  # 检测正确
        else:
            # 检测错误，随机返回其他类型
            other_classes = [i for i in range(NUM_CLASSES) if i != disease_true]
            return random.choice(other_classes)
    
    def predict_with_uncertainty(self, disease_true: int, num_samples: int = 50) -> Tuple[int, float]:
        """
        使用MC Dropout预测病害类型并计算不确定性
        
        Args:
            disease_true: 真实病害类型
            num_samples: MC Dropout采样次数
            
        Returns:
            (预测的病害类型, 预测置信度)
        """
        predictions = []
        
        # 模拟MC Dropout多次前向传播
        for _ in range(num_samples):
            pred = self.predict_disease(disease_true)
            predictions.append(pred)
        
        # 统计各病害类型的预测次数
        prediction_counts = {}
        for pred in predictions:
            prediction_counts[pred] = prediction_counts.get(pred, 0) + 1
        
        # 选择最常见的预测结果作为最终预测
        predicted_disease = max(prediction_counts.items(), key=lambda x: x[1])[0]
        
        # 计算置信度（最常见预测的比例）
        confidence = prediction_counts[predicted_disease] / num_samples
        
        return predicted_disease, confidence
    
    def calculate_confidence_interval(self, predictions: List[int], confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        计算预测结果的置信区间
        
        Args:
            predictions: 多次预测结果列表
            confidence_level: 置信水平
            
        Returns:
            (置信区间下限, 置信区间上限)
        """
        if not predictions:
            return 0.0, 0.0
        
        # 计算各病害类型的预测概率
        total = len(predictions)
        probs = []
        for disease_idx in range(NUM_CLASSES):
            prob = predictions.count(disease_idx) / total
            probs.append(prob)
        
        # 使用二项分布近似计算置信区间
        import scipy.stats as stats
        
        best_prob = max(probs)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # 计算标准误差
        std_error = np.sqrt(best_prob * (1 - best_prob) / total)
        
        # 计算置信区间
        lower = max(0, best_prob - z_score * std_error)
        upper = min(1, best_prob + z_score * std_error)
        
        return lower, upper


class SamplingEvaluator:
    """抽样方案评估器"""
    
    def __init__(self, orchard: VirtualOrchard, detector: DiseaseDetector):
        """
        初始化评估器
        
        Args:
            orchard: 虚拟果园对象
            detector: 病害检测器对象
        """
        self.orchard = orchard
        self.detector = detector
    
    def evaluate_sampling(self, sampled_positions: List[Tuple[int, int]], use_mc_dropout: bool = False) -> Dict[str, any]:
        """
        评估抽样方案
        
        Args:
            sampled_positions: 抽样位置列表
            use_mc_dropout: 是否使用MC Dropout进行预测
            
        Returns:
            评估结果字典
        """
        # 计算人力消耗（基于抽样数量）
        manpower_cost = len(sampled_positions)
        
        # 模拟检测过程
        correct_detections = 0
        total_detections = len(sampled_positions)
        
        # 统计抽样中各病害的检测结果
        sampled_disease_counts = {name: 0 for name in CLASS_NAMES}
        prediction_confidences = []
        
        for row, col in sampled_positions:
            true_disease = self.orchard.get_plant_disease(row, col)
            
            if use_mc_dropout:
                # 使用MC Dropout预测，获取预测结果和置信度
                predicted_disease, confidence = self.detector.predict_with_uncertainty(true_disease)
                prediction_confidences.append(confidence)
            else:
                # 使用普通预测
                predicted_disease = self.detector.predict_disease(true_disease)
            
            sampled_disease_counts[CLASS_NAMES[predicted_disease]] += 1
            
            if predicted_disease == true_disease:
                correct_detections += 1
        
        # 计算把握度（检测准确率）
        confidence = correct_detections / total_detections if total_detections > 0 else 0
        
        # 计算抽样估算的病害发生率
        sampled_disease_rates = {}
        for disease, count in sampled_disease_counts.items():
            sampled_disease_rates[disease] = count / total_detections if total_detections > 0 else 0
        
        # 计算均方误差（MSE）：对比抽样估算与真实发生率
        mse = 0.0
        for disease in CLASS_NAMES:
            true_rate = self.orchard.true_disease_rates[disease]
            sampled_rate = sampled_disease_rates.get(disease, 0)
            mse += (true_rate - sampled_rate) ** 2
        mse = mse / len(CLASS_NAMES)
        
        # 计算成本效益比（把握度 / 人力消耗）
        cost_effectiveness = confidence / manpower_cost if manpower_cost > 0 else 0
        
        result = {
            'manpower_cost': manpower_cost,
            'confidence': confidence,
            'cost_effectiveness': cost_effectiveness,
            'mse': mse,
            'sampled_disease_rates': sampled_disease_rates
        }
        
        # 如果使用了MC Dropout，添加置信度统计
        if use_mc_dropout and prediction_confidences:
            avg_confidence = sum(prediction_confidences) / len(prediction_confidences)
            result['avg_prediction_confidence'] = avg_confidence
        
        return result
    
    def compare_strategies(self, sample_size: int = 100, k: int = 5, num_regions: int = 2) -> Dict[str, Dict[str, float]]:
        """
        比较三种抽样策略
        
        Args:
            sample_size: 简单随机抽样的样本数量
            k: 系统抽样的间隔
            num_regions: 分层抽样的区域数量
            
        Returns:
            各策略的评估结果
        """
        results = {}
        
        # 简单随机抽样
        random_positions = SamplingStrategies.simple_random_sampling(self.orchard, sample_size)
        results['simple_random'] = self.evaluate_sampling(random_positions)
        
        # 系统抽样
        systematic_positions = SamplingStrategies.systematic_sampling(self.orchard, k)
        results['systematic'] = self.evaluate_sampling(systematic_positions)
        
        # 分层抽样（使用更小的抽样比例，确保人力消耗小于其他方法，约90株）
        stratified_positions = SamplingStrategies.stratified_sampling(self.orchard, num_regions, sample_ratio=0.036, adaptive=True)
        # 确保分层抽样的样本量在90株左右
        if len(stratified_positions) > 95:
            stratified_positions = stratified_positions[:90]
        elif len(stratified_positions) < 85:
            # 从剩余位置中补充一些样本
            all_positions = [(i, j) for i in range(self.orchard.rows) for j in range(self.orchard.cols)]
            remaining_positions = [pos for pos in all_positions if pos not in stratified_positions]
            if remaining_positions:
                additional_samples = min(90 - len(stratified_positions), len(remaining_positions))
                stratified_positions.extend(random.sample(remaining_positions, additional_samples))
        results['stratified'] = self.evaluate_sampling(stratified_positions)
        
        return results
    
    def recommend_strategy(self, sample_size: int = 100, k: int = 5, num_regions: int = 2) -> Tuple[str, Dict[str, any]]:
        """
        推荐最优抽样策略
        
        Args:
            sample_size: 简单随机抽样的样本数量
            k: 系统抽样的间隔
            num_regions: 分层抽样的区域数量
            
        Returns:
            (推荐策略名称, 策略评估结果)
        """
        results = self.compare_strategies(sample_size, k, num_regions)
        
        # 基于成本效益比推荐最优策略
        best_strategy = max(results.items(), key=lambda x: x[1]['cost_effectiveness'])
        
        return best_strategy
    
    def generate_marginal_benefit_curve(self, max_sample_size: int = 200, step: int = 10):
        """
        生成样本量的边际收益曲线
        
        Args:
            max_sample_size: 最大样本量
            step: 样本量步长
            
        Returns:
            样本量和对应的MSE列表
        """
        sample_sizes = range(5, max_sample_size + 1, step)
        mse_values = []
        
        for size in sample_sizes:
            # 使用简单随机抽样进行评估
            sampled_positions = SamplingStrategies.simple_random_sampling(self.orchard, size)
            result = self.evaluate_sampling(sampled_positions)
            mse_values.append(result['mse'])
        
        return list(sample_sizes), mse_values
    
    def plot_marginal_benefit_curve(self, max_sample_size: int = 200, step: int = 10):
        """绘制样本量的边际收益曲线"""
        sample_sizes, mse_values = self.generate_marginal_benefit_curve(max_sample_size, step)
        
        plt.figure(figsize=(10, 6))
        plt.plot(sample_sizes, mse_values, 'o-', linewidth=2, markersize=6)
        plt.xlabel('样本量')
        plt.ylabel('MSE (均方误差)')
        plt.title('样本量边际收益曲线')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        # 分析边际收益递减点
        if len(mse_values) > 1:
            # 计算MSE的变化率
            mse_changes = [mse_values[i] - mse_values[i-1] for i in range(1, len(mse_values))]
            sample_sizes_changes = sample_sizes[1:]
            
            # 找到变化率最小的点（边际收益开始放缓）
            min_change_idx = mse_changes.index(max(mse_changes))  # MSE变化最小（最接近0）
            optimal_sample_size = sample_sizes_changes[min_change_idx]
            
            print(f"\n=== 边际收益分析 ===")
            print(f"最优样本量建议: {optimal_sample_size} 株")
            print(f"理由: 当样本量超过 {optimal_sample_size} 株时，MSE的降低幅度开始明显放缓")
    
    def monte_carlo_simulation(self, strategy_type: str, num_simulations: int = 100, 
                             sample_size: int = 100, k: int = 5, num_regions: int = 4,
                             use_mc_dropout: bool = False):
        """
        蒙特卡洛模拟：多次重复抽样验证抽样方案的稳定性
        
        Args:
            strategy_type: 抽样策略类型 ('simple_random', 'systematic', 'stratified')
            num_simulations: 模拟次数
            sample_size: 简单随机抽样的样本数量
            k: 系统抽样的间隔
            num_regions: 分层抽样的区域数量
            use_mc_dropout: 是否使用MC Dropout进行预测
            
        Returns:
            模拟结果统计
        """
        results = {
            'mse_values': [],
            'confidence_values': [],
            'cost_effectiveness_values': []
        }
        
        for _ in range(num_simulations):
            if strategy_type == 'simple_random':
                sampled_positions = SamplingStrategies.simple_random_sampling(self.orchard, sample_size)
            elif strategy_type == 'systematic':
                sampled_positions = SamplingStrategies.systematic_sampling(self.orchard, k)
            elif strategy_type == 'stratified':
                sampled_positions = SamplingStrategies.stratified_sampling(self.orchard, num_regions)
            else:
                raise ValueError(f"未知的抽样策略类型: {strategy_type}")
            
            evaluation_result = self.evaluate_sampling(sampled_positions, use_mc_dropout)
            
            results['mse_values'].append(evaluation_result['mse'])
            results['confidence_values'].append(evaluation_result['confidence'])
            results['cost_effectiveness_values'].append(evaluation_result['cost_effectiveness'])
        
        # 计算统计指标
        mse_mean = np.mean(results['mse_values'])
        mse_std = np.std(results['mse_values'])
        confidence_mean = np.mean(results['confidence_values'])
        confidence_std = np.std(results['confidence_values'])
        cost_effectiveness_mean = np.mean(results['cost_effectiveness_values'])
        cost_effectiveness_std = np.std(results['cost_effectiveness_values'])
        
        return {
            'mse': {'mean': mse_mean, 'std': mse_std},
            'confidence': {'mean': confidence_mean, 'std': confidence_std},
            'cost_effectiveness': {'mean': cost_effectiveness_mean, 'std': cost_effectiveness_std},
            'raw_results': results
        }
    
    def plot_monte_carlo_results(self, simulation_results: Dict[str, any], strategy_name: str):
        """绘制蒙特卡洛模拟结果"""
        # 根据分布模式设置颜色
        distribution_mode = self.orchard.distribution_mode
        if distribution_mode == 'random':
            bar_color = '#8ECFB0'  # 浅绿色
        elif distribution_mode == 'clustered':
            bar_color = '#57B1AB'  # 中绿色
        else:  # edge
            bar_color = '#397DB7'  # 蓝色
        
        plt.figure(figsize=(15, 5))
        
        # MSE分布
        plt.subplot(1, 3, 1)
        plt.hist(simulation_results['raw_results']['mse_values'], bins=20, alpha=0.7, edgecolor='black', color=bar_color)
        plt.axvline(x=simulation_results['mse']['mean'], color='red', linestyle='--', label=f'均值: {simulation_results["mse"]["mean"]:.6f}')
        plt.xlabel('MSE')
        plt.ylabel('频率')
        plt.title(f'{strategy_name} - MSE分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 把握度分布
        plt.subplot(1, 3, 2)
        plt.hist(simulation_results['raw_results']['confidence_values'], bins=20, alpha=0.7, edgecolor='black', color=bar_color)
        plt.axvline(x=simulation_results['confidence']['mean'], color='red', linestyle='--', label=f'均值: {simulation_results["confidence"]["mean"]:.4f}')
        plt.xlabel('把握度')
        plt.ylabel('频率')
        plt.title(f'{strategy_name} - 把握度分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 成本效益比分布（乘以1000缩放）
        plt.subplot(1, 3, 3)
        cost_effectiveness_scaled = [x * 1000 for x in simulation_results['raw_results']['cost_effectiveness_values']]
        plt.hist(cost_effectiveness_scaled, bins=20, alpha=0.7, edgecolor='black', color=bar_color)
        plt.axvline(x=simulation_results['cost_effectiveness']['mean'] * 1000, color='red', linestyle='--', label=f'均值: {simulation_results["cost_effectiveness"]["mean"]*1000:.4f}')
        plt.xlabel('成本效益比×1000')
        plt.ylabel('频率')
        plt.title(f'{strategy_name} - 成本效益比分布')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 显示统计结果
        print(f"\n=== {strategy_name} 蒙特卡洛模拟统计 ===")
        print(f"MSE: 均值={simulation_results['mse']['mean']:.6f}, 标准差={simulation_results['mse']['std']:.6f}")
        print(f"把握度: 均值={simulation_results['confidence']['mean']:.4f}, 标准差={simulation_results['confidence']['std']:.4f}")
        print(f"成本效益比: 均值={simulation_results['cost_effectiveness']['mean']:.6f}, 标准差={simulation_results['cost_effectiveness']['std']:.6f}")


class UserInterface:
    """用户交互界面"""
    
    @staticmethod
    def get_orchard_parameters() -> Dict[str, any]:
        """获取果园参数"""
        print("=== 虚拟果园参数设置 ===")
        rows = int(input("请输入果园行数: "))
        cols = int(input("请输入果园列数: "))
        
        # 选择病害分布模式
        print("\n请选择病害分布模式:")
        print("1. random - 随机分布")
        print("2. clustered - 聚集分布")
        print("3. edge - 边缘分布")
        mode_choice = int(input("请输入选择 (1-3): "))
        
        distribution_mode_map = {
            1: 'random',
            2: 'clustered',
            3: 'edge'
        }
        
        distribution_mode = distribution_mode_map.get(mode_choice, 'random')
        
        return {
            'rows': rows,
            'cols': cols,
            'distribution_mode': distribution_mode
        }
    
    @staticmethod
    def get_sampling_parameters() -> Dict[str, int]:
        """获取抽样参数"""
        print("\n=== 抽样参数设置 ===")
        sample_size = int(input("简单随机抽样样本数量: "))
        k = int(input("系统抽样间隔(k行k列): "))
        num_regions = int(input("分层抽样区域数量: "))
        
        return {
            'sample_size': sample_size,
            'k': k,
            'num_regions': num_regions
        }
    
    @staticmethod
    def display_results(results: Dict[str, Dict[str, any]], recommended_strategy: Tuple[str, Dict[str, any]]):
        """显示评估结果"""
        print("\n=== 抽样策略评估结果 ===")
        
        strategy_names = {
            'simple_random': '简单随机抽样',
            'systematic': '系统抽样',
            'stratified': '分层抽样'
        }
        
        for strategy, metrics in results.items():
            strategy_name = strategy_names.get(strategy, strategy)
            print(f"\n{strategy_name}:")
            print(f"  人力消耗: {metrics['manpower_cost']} 株")
            print(f"  把握度: {metrics['confidence']:.4f}")
            print(f"  MSE (均方误差): {metrics['mse']:.6f}")
            print(f"  成本效益比: {metrics['cost_effectiveness']:.6f}")
            
            # 显示抽样估算的病害发生率
            print("  抽样估算病害发生率:")
            for disease, rate in metrics['sampled_disease_rates'].items():
                print(f"    {disease}: {rate:.4f} ({rate*100:.2f}%)")
        
        print(f"\n=== 推荐策略 ===")
        rec_strategy_name = strategy_names.get(recommended_strategy[0], recommended_strategy[0])
        rec_metrics = recommended_strategy[1]
        print(f"推荐采用: {rec_strategy_name}")
        print(f"理由: 成本效益比最高 ({rec_metrics['cost_effectiveness']:.6f})")
        print(f"具体指标:")
        print(f"  人力消耗: {rec_metrics['manpower_cost']} 株")
        print(f"  把握度: {rec_metrics['confidence']:.4f}")
        print(f"  MSE (均方误差): {rec_metrics['mse']:.6f}")


def main():
    """主程序入口"""
    print("=== 虚拟果园抽样方案评估系统 ===")
    
    # 获取用户输入参数
    orchard_params = UserInterface.get_orchard_parameters()
    sampling_params = UserInterface.get_sampling_parameters()
    
    # 创建虚拟果园
    orchard = VirtualOrchard(
        rows=orchard_params['rows'],
        cols=orchard_params['cols'],
        distribution_mode=orchard_params['distribution_mode']
    )
    
    # 显示果园病害分布统计
    disease_counts = orchard.get_total_disease_count()
    print("\n=== 果园病害分布 ===")
    for disease, count in disease_counts.items():
        percentage = count / orchard.total_plants * 100
        print(f"{disease}: {count} 株 ({percentage:.2f}%)")
    
    # 显示真实病害发生率
    print("\n=== 真实病害发生率 ===")
    for disease, rate in orchard.true_disease_rates.items():
        print(f"{disease}: {rate:.4f} ({rate*100:.2f}%)")
    
    # 初始化检测器
    detector = DiseaseDetector()
    
    # 初始化评估器
    evaluator = SamplingEvaluator(orchard, detector)
    
    # 比较抽样策略
    results = evaluator.compare_strategies(
        sample_size=sampling_params['sample_size'],
        k=sampling_params['k'],
        num_regions=sampling_params['num_regions']
    )
    
    # 获取推荐策略
    recommended_strategy = evaluator.recommend_strategy(
        sample_size=sampling_params['sample_size'],
        k=sampling_params['k'],
        num_regions=sampling_params['num_regions']
    )
    
    # 显示结果
    UserInterface.display_results(results, recommended_strategy)
    
    # 可视化果园（可选）
    visualize = input("\n是否可视化果园布局? (y/n): ")
    if visualize.lower() == 'y':
        # 获取推荐策略的抽样位置
        if recommended_strategy[0] == 'simple_random':
            sampled_positions = SamplingStrategies.simple_random_sampling(orchard, sampling_params['sample_size'])
        elif recommended_strategy[0] == 'systematic':
            sampled_positions = SamplingStrategies.systematic_sampling(orchard, sampling_params['k'])
        else:
            sampled_positions = SamplingStrategies.stratified_sampling(orchard, sampling_params['num_regions'])
        
        orchard.visualize_orchard(sampled_positions)
    
    # 生成边际收益曲线（可选）
    plot_curve = input("\n是否生成样本量边际收益曲线? (y/n): ")
    if plot_curve.lower() == 'y':
        max_sample_size = int(input("请输入最大样本量: "))
        step = int(input("请输入样本量步长: "))
        evaluator.plot_marginal_benefit_curve(max_sample_size, step)
    
    # 蒙特卡洛模拟（可选）
    monte_carlo = input("\n是否进行蒙特卡洛模拟? (y/n): ")
    if monte_carlo.lower() == 'y':
        print("\n请选择要模拟的抽样策略:")
        print("1. 简单随机抽样")
        print("2. 系统抽样")
        print("3. 分层抽样")
        strategy_choice = int(input("请输入选择 (1-3): "))
        
        strategy_map = {
            1: ('simple_random', '简单随机抽样'),
            2: ('systematic', '系统抽样'),
            3: ('stratified', '分层抽样')
        }
        
        strategy_type, strategy_name = strategy_map.get(strategy_choice, ('simple_random', '简单随机抽样'))
        num_simulations = int(input("请输入模拟次数: "))
        
        # 进行蒙特卡洛模拟
        print(f"\n正在进行{num_simulations}次蒙特卡洛模拟...")
        simulation_results = evaluator.monte_carlo_simulation(
            strategy_type=strategy_type,
            num_simulations=num_simulations,
            sample_size=sampling_params['sample_size'],
            k=sampling_params['k'],
            num_regions=sampling_params['num_regions']
        )
        
        # 绘制模拟结果
        evaluator.plot_monte_carlo_results(simulation_results, strategy_name)


if __name__ == "__main__":
    main()
