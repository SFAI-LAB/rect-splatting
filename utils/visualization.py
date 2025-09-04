"""
3D Histogram visualization utilities for p-norm distribution
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import torch
import threading
from collections import deque

class Simple2DHistogramVisualizer:
    """간단한 2D 히스토그램 시각화 (실시간 업데이트)"""
    
    def __init__(self, max_history=200, update_interval=50, p_min=1.0, p_max=10.0, num_bins=25):
        self.max_history = max_history
        self.update_interval = update_interval
        
        # 데이터 저장용
        self.iterations = deque(maxlen=max_history)
        self.p_distributions = deque(maxlen=max_history)
        
        # p값 범위 설정 (동적으로 조정됨)
        self.p_min = p_min
        self.p_max = p_max  # 초기값, 실제 데이터에 따라 조정
        self.num_bins = num_bins
        self.p_bins = np.linspace(self.p_min, self.p_max, self.num_bins)
        self.p_centers = (self.p_bins[:-1] + self.p_bins[1:]) / 2
        
        # 플롯 설정
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.lines = []
        
        # 스레드 안전을 위한 락
        self.data_lock = threading.Lock()
        
    def add_data(self, iteration, p_values):
        """데이터 추가"""
        with self.data_lock:
            if isinstance(p_values, torch.Tensor):
                p_values_np = p_values.detach().cpu().numpy()
            else:
                p_values_np = p_values
            
            # p값 범위 동적 업데이트
            current_min = max(1.0, np.min(p_values_np))  # 최소값은 1.0으로 고정
            current_max = 5 #np.max(p_values_np)
            
            # 범위가 변경되었다면 bins 업데이트
            if current_max > self.p_max or (current_max < self.p_max * 0.8 and current_max > 5.0):
                self.p_max = max(current_max * 1.1, 5.0)  # 약간의 마진 추가, 최소 5.0
                self.p_bins = np.linspace(self.p_min, self.p_max, self.num_bins)
                self.p_centers = (self.p_bins[:-1] + self.p_bins[1:]) / 2
                
            # 히스토그램 계산
            counts, _ = np.histogram(p_values_np, bins=self.p_bins)
            normalized_counts = counts / len(p_values_np)
            
            self.iterations.append(iteration)
            self.p_distributions.append(normalized_counts)
            
    def should_update(self, iteration):
        return iteration % self.update_interval == 0
        
    def initialize_plot(self):
        """플롯 초기화"""
        if self.fig is not None:
            return
            
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 첫 번째: 현재 분포
        self.ax1.set_title('Current P-norm Distribution', fontsize=14)
        self.ax1.set_xlabel('P-norm Value', fontsize=12)
        self.ax1.set_ylabel('Proportion', fontsize=12)
        self.ax1.grid(True, alpha=0.3)
        
        # 두 번째: 시간에 따른 변화 (히트맵)
        self.ax2.set_title('P-norm Distribution Evolution', fontsize=14)
        self.ax2.set_xlabel('P-norm Value', fontsize=12)
        self.ax2.set_ylabel('Iteration', fontsize=12)
        
        plt.tight_layout()
        
    def update_plot(self):
        """플롯 업데이트"""
        if len(self.iterations) == 0:
            return
            
        with self.data_lock:
            if self.fig is None:
                self.initialize_plot()
                
            # 첫 번째 플롯 클리어
            self.ax1.clear()
            self.ax2.clear()
            
            iterations = list(self.iterations)
            distributions = list(self.p_distributions)
            
            if len(iterations) == 0:
                return
                
            # 현재 분포 (막대 그래프)
            current_dist = distributions[-1] if distributions else np.zeros_like(self.p_centers)
            bars = self.ax1.bar(self.p_centers, current_dist, width=self.p_centers[1]-self.p_centers[0], 
                               alpha=0.7, color='skyblue', edgecolor='navy', linewidth=0.5)
            
            # 특별한 p값 영역 표시
            self.ax1.axvline(x=1.5, color='red', linestyle='--', alpha=0.7, label='L1-L2 boundary')
            self.ax1.axvline(x=3.0, color='orange', linestyle='--', alpha=0.7, label='L2-L∞ boundary')
            
            self.ax1.set_title(f'Current P-norm Distribution (Iter: {iterations[-1]})', fontsize=14)
            self.ax1.set_xlabel('P-norm Value', fontsize=12)
            self.ax1.set_ylabel('Proportion', fontsize=12)
            self.ax1.grid(True, alpha=0.3)
            self.ax1.legend()
            
            # 시간에 따른 변화 (히트맵)
            if len(distributions) > 1:
                dist_array = np.array(distributions)
                im = self.ax2.imshow(dist_array.T, aspect='auto', origin='lower', 
                                   extent=[iterations[0], iterations[-1], self.p_min, self.p_max], 
                                   cmap='viridis', interpolation='bilinear')
                
                # 컬러바
                if hasattr(self, 'colorbar'):
                    self.colorbar.remove()
                self.colorbar = plt.colorbar(im, ax=self.ax2)
                self.colorbar.set_label('Proportion', fontsize=12)
                
            self.ax2.set_title('P-norm Distribution Evolution', fontsize=14)
            self.ax2.set_xlabel('Iteration', fontsize=12)
            self.ax2.set_ylabel('P-norm Value', fontsize=12)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
            
    def save_final_plot(self, save_path):
        """최종 플롯을 파일로 저장 - 마지막 iteration의 2D 히스토그램만"""
        if len(self.iterations) == 0:
            print("No data to save")
            return
            
        with self.data_lock:
            # 새로운 figure 생성 (2D 히스토그램만)
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            iterations = list(self.iterations)
            distributions = list(self.p_distributions)
            
            if len(distributions) == 0:
                print("No distribution data to save")
                return
                
            # 마지막 iteration의 분포만 사용
            final_dist = distributions[-1]
            final_iter = iterations[-1]
            
            # 막대 그래프로 표시
            bars = ax.bar(self.p_centers, final_dist, 
                         width=self.p_centers[1]-self.p_centers[0], 
                         alpha=0.7, color='skyblue', edgecolor='navy', linewidth=0.8)
            
            # 특별한 p값 영역 표시
            ax.axvline(x=1.5, color='red', linestyle='--', alpha=0.7, label='L1-L2 boundary')
            ax.axvline(x=3.0, color='orange', linestyle='--', alpha=0.7, label='L2-L∞ boundary')
            
            # 제목과 레이블 설정
            ax.set_title(f'Final P-norm Distribution (Iteration: {final_iter})', fontsize=16, fontweight='bold')
            ax.set_xlabel('P-norm Value', fontsize=14)
            ax.set_ylabel('Proportion', fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=12)
            
            # p값 범위 표시
            ax.set_xlim(self.p_min, self.p_max)
            
            # 통계 정보 추가
            max_prop = max(final_dist)
            max_p_idx = np.argmax(final_dist)
            peak_p = self.p_centers[max_p_idx]
            
            # 텍스트 박스로 통계 정보 표시
            stats_text = f'Peak at p={peak_p:.2f}\nMax proportion: {max_prop:.3f}\nTotal bins: {len(final_dist)}'
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)  # 메모리 정리
            
            print(f"Final 2D histogram plot saved to: {save_path}")
            
    def close(self):
        """리소스 정리"""
        if self.fig is not None:
            plt.close(self.fig)
            plt.ioff()

class RealTimeHistogramVisualizer:
    def __init__(self, max_history=100, update_interval=100, p_min=1.0, p_max=10.0, num_bins=32, store_all_data=True):
        """
        실시간 p값 분포 3D 히스토그램 시각화
        
        Args:
            max_history: 저장할 최대 히스토리 개수 (store_all_data=False일 때만 적용)
            update_interval: 히스토그램 업데이트 간격 (iterations)
            p_min: P-norm 최소값
            p_max: P-norm 초기 최대값 (동적으로 조정됨)
            num_bins: 히스토그램 빈 개수
            store_all_data: 전체 iteration 데이터를 저장할지 여부
        """
        self.max_history = max_history
        self.update_interval = update_interval
        self.store_all_data = store_all_data
        
        # 데이터 저장용
        if store_all_data:
            # 전체 데이터 저장 (리스트 사용)
            self.iterations = []
            self.normalized_distributions = []
            self.raw_distributions = []
        else:
            # 제한된 히스토리 저장 (deque 사용)
            self.iterations = deque(maxlen=max_history)
            self.normalized_distributions = deque(maxlen=max_history)
            self.raw_distributions = deque(maxlen=max_history)
        
        self.p_distributions = deque(maxlen=max_history)  # 이건 항상 deque 사용
        
        # p값 범위 설정 (동적으로 조정됨)
        self.p_min = p_min
        self.p_max = p_max  # 초기값
        self.num_bins = num_bins
        self.p_bins = np.linspace(self.p_min, self.p_max, self.num_bins)  # 히스토그램 빈
        
        # 플롯 설정
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        self.is_plotting = False
        
        # 스레드 안전을 위한 락
        self.data_lock = threading.Lock()
        
    def add_data(self, iteration, p_values):
        """
        새로운 iteration의 p값 분포 데이터 추가
        
        Args:
            iteration: 현재 iteration
            p_values: torch.Tensor, p값들
        """
        with self.data_lock:
            if isinstance(p_values, torch.Tensor):
                p_values_np = p_values.detach().cpu().numpy()
            else:
                p_values_np = p_values
            
            # p값 범위 동적 업데이트
            current_min = max(1.0, np.min(p_values_np))
            current_max = 5 #np.max(p_values_np)
            
            # 범위가 변경되었다면 bins 업데이트
            if current_max > self.p_max or (current_max < self.p_max * 0.8 and current_max > 5.0):
                self.p_max = max(current_max * 1.1, 5.0)  # 약간의 마진 추가
                self.p_bins = np.linspace(self.p_min, self.p_max, self.num_bins)
                
            # 히스토그램 계산
            counts, _ = np.histogram(p_values_np, bins=self.p_bins)
            normalized_counts = counts / len(p_values_np)  # 정규화 (비율)
            
            # 데이터 저장 방식 선택
            if self.store_all_data:
                self.iterations.append(iteration)
                self.raw_distributions.append(counts)
                self.normalized_distributions.append(normalized_counts)
            else:
                self.iterations.append(iteration)
                self.raw_distributions.append(counts)
                self.normalized_distributions.append(normalized_counts)
            
    def should_update(self, iteration):
        """업데이트 여부 판단"""
        return iteration % self.update_interval == 0 or iteration <= 1000
        
    def initialize_plot(self):
        """플롯 초기화"""
        if self.fig is not None:
            return
            
        plt.ion()  # 인터랙티브 모드 활성화
        self.fig = plt.figure(figsize=(16, 8))
        
        # 첫 번째 서브플롯: 정규화된 분포
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax1.set_title('Normalized P-norm Distribution (Proportions)', fontsize=14)
        self.ax1.set_xlabel('P-norm Value', fontsize=12)
        self.ax1.set_ylabel('Iteration', fontsize=12)
        self.ax1.set_zlabel('Proportion', fontsize=12)
        
        # 두 번째 서브플롯: 원시 개수 분포
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        self.ax2.set_title('Raw P-norm Distribution (Counts)', fontsize=14)
        self.ax2.set_xlabel('P-norm Value', fontsize=12)
        self.ax2.set_ylabel('Iteration', fontsize=12)
        self.ax2.set_zlabel('Count', fontsize=12)
        
        plt.tight_layout()
        
    def update_plot(self):
        """플롯 업데이트"""
        if len(self.iterations) == 0:
            return
            
        with self.data_lock:
            if self.fig is None:
                self.initialize_plot()
                
            # 기존 플롯 클리어
            self.ax1.clear()
            self.ax2.clear()
            
            # 데이터 준비
            iterations = list(self.iterations)
            normalized_dists = list(self.normalized_distributions)
            raw_dists = list(self.raw_distributions)
            
            if len(iterations) == 0:
                return
            
            # 전체 데이터를 사용하되, 너무 많으면 샘플링
            if self.store_all_data and len(iterations) > 200:
                # 너무 많은 데이터가 있으면 균등하게 샘플링
                step = max(1, len(iterations) // 200)
                iterations = iterations[::step]
                normalized_dists = normalized_dists[::step]
                raw_dists = raw_dists[::step]
                
            # X, Y 좌표 생성 (동적 범위 사용)
            p_centers = (self.p_bins[:-1] + self.p_bins[1:]) / 2
            X, Y = np.meshgrid(p_centers, iterations)
            
            # Z 데이터 준비
            Z_normalized = np.array(normalized_dists)
            Z_raw = np.array(raw_dists)
            
            # 첫 번째 플롯: 정규화된 분포
            self.ax1.plot_surface(X, Y, Z_normalized, cmap=cm.viridis, alpha=0.8)
            total_iters = len(self.iterations)
            shown_iters = len(iterations)
            self.ax1.set_title(f'Normalized P-norm Distribution (Showing {shown_iters}/{total_iters} iters)', fontsize=14)
            self.ax1.set_xlabel('P-norm Value', fontsize=12)
            self.ax1.set_ylabel('Iteration', fontsize=12)
            self.ax1.set_zlabel('Proportion', fontsize=12)
            
            # 두 번째 플롯: 원시 개수 분포
            self.ax2.plot_surface(X, Y, Z_raw, cmap=cm.plasma, alpha=0.8)
            self.ax2.set_title(f'Raw P-norm Distribution (Showing {shown_iters}/{total_iters} iters)', fontsize=14)
            self.ax2.set_xlabel('P-norm Value', fontsize=12)
            self.ax2.set_ylabel('Iteration', fontsize=12)
            self.ax2.set_zlabel('Count', fontsize=12)
            
            # 뷰 각도 설정
            self.ax1.view_init(elev=20, azim=45)
            self.ax2.view_init(elev=20, azim=45)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)  # 짧은 일시정지로 화면 업데이트
            
    def plot_wireframe_style(self):
        """와이어프레임 스타일로 플롯 업데이트"""
        if len(self.iterations) == 0:
            return
            
        with self.data_lock:
            if self.fig is None:
                self.initialize_plot()
                
            # 기존 플롯 클리어
            self.ax1.clear()
            self.ax2.clear()
            
            # 데이터 준비
            iterations = list(self.iterations)
            normalized_dists = list(self.normalized_distributions)
            raw_dists = list(self.raw_distributions)
            
            if len(iterations) == 0:
                return
                
            # X, Y 좌표 생성
            p_centers = (self.p_bins[:-1] + self.p_bins[1:]) / 2
            X, Y = np.meshgrid(p_centers, iterations)
            
            # Z 데이터 준비
            Z_normalized = np.array(normalized_dists)
            Z_raw = np.array(raw_dists)
            
            # 첫 번째 플롯: 정규화된 분포 (와이어프레임)
            self.ax1.plot_wireframe(X, Y, Z_normalized, color='blue', alpha=0.6, linewidth=0.5)
            self.ax1.set_title('Normalized P-norm Distribution (Proportions)', fontsize=14)
            self.ax1.set_xlabel('P-norm Value', fontsize=12)
            self.ax1.set_ylabel('Iteration', fontsize=12)
            self.ax1.set_zlabel('Proportion', fontsize=12)
            
            # 두 번째 플롯: 원시 개수 분포 (와이어프레임)
            self.ax2.plot_wireframe(X, Y, Z_raw, color='red', alpha=0.6, linewidth=0.5)
            self.ax2.set_title('Raw P-norm Distribution (Counts)', fontsize=14)
            self.ax2.set_xlabel('P-norm Value', fontsize=12)
            self.ax2.set_ylabel('Iteration', fontsize=12)
            self.ax2.set_zlabel('Count', fontsize=12)
            
            # 뷰 각도 설정
            self.ax1.view_init(elev=30, azim=60)
            self.ax2.view_init(elev=30, azim=60)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
            
    def save_final_plot(self, save_path):
        """최종 플롯을 파일로 저장"""
        if self.fig is not None:
            # 최종 저장 시에는 전체 데이터 사용
            self.plot_final_full_data()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Final histogram plot saved to: {save_path}")
            
    def plot_final_full_data(self):
        """최종 저장용: 전체 데이터를 사용하여 플롯 생성"""
        if len(self.iterations) == 0:
            return
            
        with self.data_lock:
            if self.fig is None:
                self.initialize_plot()
                
            # 기존 플롯 클리어
            self.ax1.clear()
            self.ax2.clear()
            
            # 전체 데이터 사용
            iterations = list(self.iterations)
            normalized_dists = list(self.normalized_distributions)
            raw_dists = list(self.raw_distributions)
            
            if len(iterations) == 0:
                return
                
            print(f"Plotting final histogram with {len(iterations)} iterations...")
            
            # 데이터가 너무 많으면 적절히 샘플링 (하지만 더 많이 보여줌)
            if len(iterations) > 500:
                step = max(1, len(iterations) // 500)
                iterations = iterations[::step]
                normalized_dists = normalized_dists[::step]
                raw_dists = raw_dists[::step]
                print(f"Downsampled to {len(iterations)} points for visualization")
                
            # X, Y 좌표 생성
            p_centers = (self.p_bins[:-1] + self.p_bins[1:]) / 2
            X, Y = np.meshgrid(p_centers, iterations)
            
            # Z 데이터 준비
            Z_normalized = np.array(normalized_dists)
            Z_raw = np.array(raw_dists)
            
            # 첫 번째 플롯: 정규화된 분포
            self.ax1.plot_surface(X, Y, Z_normalized, cmap=cm.viridis, alpha=0.8)
            total_iters = len(self.iterations)
            self.ax1.set_title(f'Final Normalized P-norm Distribution ({total_iters} total iters)', fontsize=14)
            self.ax1.set_xlabel('P-norm Value', fontsize=12)
            self.ax1.set_ylabel('Iteration', fontsize=12)
            self.ax1.set_zlabel('Proportion', fontsize=12)
            
            # 두 번째 플롯: 원시 개수 분포
            self.ax2.plot_surface(X, Y, Z_raw, cmap=cm.plasma, alpha=0.8)
            self.ax2.set_title(f'Final Raw P-norm Distribution ({total_iters} total iters)', fontsize=14)
            self.ax2.set_xlabel('P-norm Value', fontsize=12)
            self.ax2.set_ylabel('Iteration', fontsize=12)
            self.ax2.set_zlabel('Count', fontsize=12)
            
            # 뷰 각도 설정
            self.ax1.view_init(elev=20, azim=45)
            self.ax2.view_init(elev=20, azim=45)
            
            plt.tight_layout()
            
    def close(self):
        """리소스 정리"""
        if self.fig is not None:
            plt.close(self.fig)
            plt.ioff()

class BarHistogramVisualizer:
    """바 히스토그램 스타일 시각화 (더 명확한 분포 표현)"""
    
    def __init__(self, max_history=50, update_interval=100, p_min=1.0, p_max=10.0, num_bins=20, store_all_data=False):
        self.max_history = max_history
        self.update_interval = update_interval
        self.store_all_data = store_all_data
        
        # 데이터 저장용
        if store_all_data:
            self.iterations = []
            self.normalized_distributions = []
            self.raw_distributions = []
        else:
            self.iterations = deque(maxlen=max_history)
            self.normalized_distributions = deque(maxlen=max_history)
            self.raw_distributions = deque(maxlen=max_history)
        
        # p값 범위 설정 (동적으로 조정됨)
        self.p_min = p_min
        self.p_max = p_max  # 초기값
        self.num_bins = num_bins
        self.p_bins = np.linspace(self.p_min, self.p_max, self.num_bins)  # 히스토그램 빈
        
        # 플롯 설정
        self.fig = None
        self.ax1 = None
        self.ax2 = None
        
        # 스레드 안전을 위한 락
        self.data_lock = threading.Lock()
        
    def add_data(self, iteration, p_values):
        """데이터 추가"""
        with self.data_lock:
            if isinstance(p_values, torch.Tensor):
                p_values_np = p_values.detach().cpu().numpy()
            else:
                p_values_np = p_values
            
            # p값 범위 동적 업데이트
            current_min = max(1.0, np.min(p_values_np))
            current_max = 5 #np.max(p_values_np)
            
            # 범위가 변경되었다면 bins 업데이트
            if current_max > self.p_max or (current_max < self.p_max * 0.8 and current_max > 5.0):
                self.p_max = max(current_max * 1.1, 5.0)
                self.p_bins = np.linspace(self.p_min, self.p_max, self.num_bins)
                
            # 히스토그램 계산
            counts, _ = np.histogram(p_values_np, bins=self.p_bins)
            normalized_counts = counts / len(p_values_np)
            
            self.iterations.append(iteration)
            self.raw_distributions.append(counts)
            self.normalized_distributions.append(normalized_counts)
            
    def should_update(self, iteration):
        return iteration % self.update_interval == 0
        
    def initialize_plot(self):
        """플롯 초기화"""
        if self.fig is not None:
            return
            
        plt.ion()
        self.fig = plt.figure(figsize=(18, 8))
        
        # 3D 바 히스토그램 플롯
        self.ax1 = self.fig.add_subplot(121, projection='3d')
        self.ax1.set_title('Normalized P-norm Distribution (3D Bar)', fontsize=14)
        
        self.ax2 = self.fig.add_subplot(122, projection='3d')
        self.ax2.set_title('Raw P-norm Distribution (3D Bar)', fontsize=14)
        
        plt.tight_layout()
        
    def update_plot(self):
        """3D 바 히스토그램으로 플롯 업데이트"""
        if len(self.iterations) == 0:
            return
            
        with self.data_lock:
            if self.fig is None:
                self.initialize_plot()
                
            self.ax1.clear()
            self.ax2.clear()
            
            iterations = list(self.iterations)
            normalized_dists = list(self.normalized_distributions)
            raw_dists = list(self.raw_distributions)
            
            if len(iterations) == 0:
                return
                
            # 바 히스토그램을 위한 좌표 설정
            p_centers = (self.p_bins[:-1] + self.p_bins[1:]) / 2
            p_width = p_centers[1] - p_centers[0]
            iter_width = 50  # iteration 간격
            
            # 각 iteration과 p값에 대한 바 그리기
            for i, (iteration, norm_dist, raw_dist) in enumerate(zip(iterations, normalized_dists, raw_dists)):
                y_pos = [iteration] * len(p_centers)
                
                # 정규화된 분포
                self.ax1.bar3d(p_centers, y_pos, [0]*len(p_centers), 
                              [p_width]*len(p_centers), [iter_width]*len(p_centers), norm_dist,
                              alpha=0.7, color=plt.cm.viridis(i/len(iterations)))
                
                # 원시 개수 분포
                self.ax2.bar3d(p_centers, y_pos, [0]*len(p_centers),
                              [p_width]*len(p_centers), [iter_width]*len(p_centers), raw_dist,
                              alpha=0.7, color=plt.cm.plasma(i/len(iterations)))
            
            # 축 설정
            self.ax1.set_xlabel('P-norm Value')
            self.ax1.set_ylabel('Iteration')
            self.ax1.set_zlabel('Proportion')
            self.ax1.set_title('Normalized P-norm Distribution (3D Bar)')
            
            self.ax2.set_xlabel('P-norm Value')
            self.ax2.set_ylabel('Iteration')
            self.ax2.set_zlabel('Count')
            self.ax2.set_title('Raw P-norm Distribution (3D Bar)')
            
            # 뷰 각도 설정
            self.ax1.view_init(elev=25, azim=45)
            self.ax2.view_init(elev=25, azim=45)
            
            plt.tight_layout()
            plt.draw()
            plt.pause(0.01)
            
    def close(self):
        """리소스 정리"""
        if self.fig is not None:
            plt.close(self.fig)
            plt.ioff()
