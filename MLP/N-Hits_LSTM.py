# import pandas as pd
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split
# import json
# import glob
# import re
# import os
# from datetime import datetime, timedelta
# from tqdm import tqdm
# import warnings
# warnings.filterwarnings('ignore')

# print("🎯 친구 스타일 N-HiTS + Hurdle Model v1.0")
# print("=" * 60)
# print("친구의 단순 전처리 + Hurdle Loss + 정확한 N-HiTS")

# # 친구 파일 경로
# FRIEND_DATA_PATH = '/home/droppgs/lg-aimers-demand-forecasting/lstm_project/joonsik'

# # 데이콘 공식 매장별 가중치
# RESTAURANT_WEIGHTS = {
#     '미라시아': 7.71,
#     '담하': 6.51,
#     '연회장': 3.48,
#     '라그로타': 3.44,
#     '느티나무 셀프BBQ': 2.78,
#     '화담숲주막': 1.43,
#     '카페테리아': 1.31,
#     '화담숲카페': 1.14,
#     '포레스트릿': 1.00
# }

# class FriendStylePreprocessor:
#     """친구 스타일 전처리기 - 단순하지만 효과적"""
    
#     def __init__(self):
#         self.label_encoders = {}
#         self.menu_categories = {}
        
#     def load_friend_data(self):
#         """친구의 전처리된 데이터 로드"""
#         preprocess_path = os.path.join(FRIEND_DATA_PATH, 'data/final_train_datasets/preprocess_train.csv')
        
#         try:
#             df = pd.read_csv(preprocess_path)
#             print(f"✅ 친구 전처리 데이터 로드: {len(df):,}개")
#             return df
#         except FileNotFoundError:
#             print("❌ 친구 파일 없음 - 기본 데이터 사용")
#             return None
    
#     def load_menu_prices(self):
#         """친구의 메뉴 가격 정보 로드"""
#         price_path = os.path.join(FRIEND_DATA_PATH, 'data/preprocessing_artifacts/outer/menu_price.json')
        
#         try:
#             with open(price_path, 'r', encoding='utf-8') as f:
#                 menu_data = json.load(f)
            
#             # 메뉴 카테고리 정보 추출
#             categories = {}
#             prices = {}
            
#             for restaurant, menus in menu_data.items():
#                 for menu_info in menus:
#                     menu_name = menu_info['name']
#                     full_name = f"{restaurant}_{menu_name}"
                    
#                     # 카테고리 정보
#                     if 'category' in menu_info:
#                         categories[full_name] = menu_info['category']
#                     else:
#                         # 메뉴명으로 카테고리 추론
#                         if any(drink in menu_name.lower() for drink in ['스프라이트', '콜라', '아이스티', '에이드', '음료']):
#                             categories[full_name] = '음료'
#                         elif any(alcohol in menu_name.lower() for alcohol in ['맥주', '소주', '막걸리', '와인', '하이볼', '칵테일']):
#                             categories[full_name] = '주류'
#                         elif any(food in menu_name.lower() for food in ['커피', '아메리카노', '라떼']):
#                             categories[full_name] = '커피'
#                         else:
#                             categories[full_name] = '음식'
                    
#                     # 가격 정보
#                     if menu_info.get('price'):
#                         try:
#                             prices[full_name] = float(menu_info['price'])
#                         except:
#                             pass
            
#             self.menu_categories = categories
#             print(f"✅ 메뉴 카테고리 정보: {len(categories)}개")
#             print(f"   음료: {sum(1 for c in categories.values() if c == '음료')}개")
#             print(f"   주류: {sum(1 for c in categories.values() if c == '주류')}개")
#             print(f"   커피: {sum(1 for c in categories.values() if c == '커피')}개")
#             print(f"   음식: {sum(1 for c in categories.values() if c == '음식')}개")
            
#             return categories, prices
            
#         except FileNotFoundError:
#             print("❌ 메뉴 가격 파일 없음")
#             return {}, {}
    
#     def create_friend_features(self, df):
#         """친구 스타일 피처 생성"""
#         df = df.copy()
        
#         # 친구 전처리 데이터가 있으면 사용
#         friend_df = self.load_friend_data()
#         if friend_df is not None:
#             # 친구의 피처들 병합
#             friend_features = ['요일', '주중', '주말', '공휴일']
#             df = df.merge(
#                 friend_df[['영업일자', '영업장명_메뉴명'] + friend_features],
#                 on=['영업일자', '영업장명_메뉴명'],
#                 how='left'
#             )
#         else:
#             # 기본 피처 생성
#             df['영업일자'] = pd.to_datetime(df['영업일자'])
#             df['요일'] = df['영업일자'].dt.day_name()
#             df['주중'] = (df['영업일자'].dt.weekday < 5).astype(int)
#             df['주말'] = (df['영업일자'].dt.weekday >= 5).astype(int)
#             df['공휴일'] = 0  # 간단히 처리
        
#         # 메뉴 카테고리 정보 추가
#         categories, prices = self.load_menu_prices()
#         df['menu_category'] = df['영업장명_메뉴명'].map(categories).fillna('음식')
        
#         # 영업장명과 메뉴명 분리
#         df['restaurant'] = df['영업장명_메뉴명'].str.split('_').str[0]
#         df['menu'] = df['영업장명_메뉴명'].str.split('_', n=1).str[1]
        
#         # 매장별 가중치
#         df['restaurant_weight'] = df['restaurant'].map(RESTAURANT_WEIGHTS).fillna(1.0)
        
#         # 기본 시간 피처
#         df['영업일자'] = pd.to_datetime(df['영업일자'])
#         df['month'] = df['영업일자'].dt.month
#         df['day'] = df['영업일자'].dt.day
#         df['dayofweek'] = df['영업일자'].dt.dayofweek
        
#         return df
    
#     def create_lag_features(self, df, is_train=True):
#         """친구 스타일 지연 피처 - 매우 단순"""
#         df = df.copy().sort_values(['영업장명_메뉴명', '영업일자'])
        
#         # 간단한 지연 피처 (7, 14일만)
#         for lag in [7, 14]:
#             df[f'lag_{lag}'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(lag)
        
#         # 간단한 이동평균 (7일만)
#         df['ma_7'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(1).rolling(
#             window=7, min_periods=1
#         ).mean()
        
#         # 결측값을 0으로 처리 (친구 스타일)
#         lag_cols = [col for col in df.columns if col.startswith(('lag_', 'ma_'))]
#         for col in lag_cols:
#             df[col] = df[col].fillna(0)
        
#         return df
    
#     def encode_features(self, df, is_train=True):
#         """피처 인코딩"""
#         categorical_features = ['요일', 'menu_category', 'restaurant', 'menu']
        
#         for col in categorical_features:
#             if col in df.columns:
#                 if is_train:
#                     self.label_encoders[col] = LabelEncoder()
#                     df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
#                 else:
#                     if col in self.label_encoders:
#                         # 새로운 카테고리는 0으로 처리
#                         known_categories = set(self.label_encoders[col].classes_)
#                         df[f'{col}_mapped'] = df[col].astype(str).apply(
#                             lambda x: x if x in known_categories else self.label_encoders[col].classes_[0]
#                         )
#                         df[f'{col}_encoded'] = self.label_encoders[col].transform(df[f'{col}_mapped'])
#                     else:
#                         df[f'{col}_encoded'] = 0
        
#         return df

# class HurdleDataset(Dataset):
#     """Hurdle Model용 데이터셋"""
    
#     def __init__(self, sequences, features, zero_targets, positive_targets):
#         self.sequences = torch.FloatTensor(sequences)
#         self.features = torch.FloatTensor(features)
#         self.zero_targets = torch.FloatTensor(zero_targets)
#         self.positive_targets = torch.FloatTensor(positive_targets)
    
#     def __len__(self):
#         return len(self.sequences)
    
#     def __getitem__(self, idx):
#         return (self.sequences[idx], self.features[idx], 
#                 self.zero_targets[idx], self.positive_targets[idx])

# class NHiTSHurdleModel(nn.Module):
#     """N-HiTS + Hurdle Model"""
    
#     def __init__(self, input_size, feature_size, hidden_size=64, num_stacks=3):
#         super(NHiTSHurdleModel, self).__init__()
        
#         self.hidden_size = hidden_size
#         self.num_stacks = num_stacks
        
#         # Feature embedding
#         self.feature_fc = nn.Sequential(
#             nn.Linear(feature_size, hidden_size),
#             nn.ReLU(),
#             nn.Dropout(0.1)
#         )
        
#         # N-HiTS Stacks (Multi-rate sampling)
#         self.stacks = nn.ModuleList()
#         for i in range(num_stacks):
#             stack = nn.ModuleDict({
#                 'linear1': nn.Linear(28 + hidden_size, hidden_size),
#                 'linear2': nn.Linear(hidden_size, hidden_size),
#                 'theta_f': nn.Linear(hidden_size, 7),  # Forecast
#                 'theta_b': nn.Linear(hidden_size, 28)  # Backcast
#             })
#             self.stacks.append(stack)
        
#         # Hurdle Components
#         # 1. Zero probability (Binary Classification)
#         self.zero_classifier = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_size // 2, 7),
#             nn.Sigmoid()  # 0이 될 확률
#         )
        
#         # 2. Positive value predictor (Regression)
#         self.positive_predictor = nn.Sequential(
#             nn.Linear(hidden_size, hidden_size // 2),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_size // 2, 7),
#             nn.Softplus()  # 양수 보장
#         )
    
#     def forward(self, sequences, features):
#         batch_size = sequences.size(0)
        
#         # Feature embedding
#         feature_emb = self.feature_fc(features)
        
#         # N-HiTS processing
#         residual = sequences.squeeze(-1)  # (batch, 28)
#         forecast_sum = torch.zeros(batch_size, 7, device=sequences.device)
        
#         for stack in self.stacks:
#             # Combine sequence and features
#             combined = torch.cat([residual, feature_emb], dim=1)
            
#             # Stack processing
#             h1 = torch.relu(stack['linear1'](combined))
#             h2 = torch.relu(stack['linear2'](h1))
            
#             # Forecast and backcast
#             forecast = stack['theta_f'](h2)
#             backcast = stack['theta_b'](h2)
            
#             # Update
#             forecast_sum = forecast_sum + forecast
#             residual = residual - backcast
        
#         # Final representation
#         final_repr = h2  # Last hidden state
        
#         # Hurdle components
#         zero_prob = self.zero_classifier(final_repr)  # P(Y = 0)
#         positive_values = self.positive_predictor(final_repr)  # E[Y | Y > 0]
        
#         return zero_prob, positive_values, forecast_sum

# class FriendStyleNHiTS:
#     """친구 스타일 N-HiTS + Hurdle 예측기"""
    
#     def __init__(self, lookback_window=28):
#         self.lookback_window = lookback_window
#         self.preprocessor = FriendStylePreprocessor()
#         self.models = {}
#         self.scalers = {}
#         self.feature_cols = []
        
#     def prepare_hurdle_data(self, df):
#         """Hurdle Model용 데이터 준비"""
#         sequences = []
#         features = []
#         zero_targets = []
#         positive_targets = []
        
#         for menu_name, group in df.groupby('영업장명_메뉴명'):
#             group = group.sort_values('영업일자').reset_index(drop=True)
            
#             if len(group) < self.lookback_window + 7:
#                 continue
            
#             for i in range(self.lookback_window, len(group) - 6):
#                 # 입력 시퀀스 (28일)
#                 seq = group['매출수량'].iloc[i-self.lookback_window:i].values
                
#                 # 특성 (마지막 날짜)
#                 feat = group[self.feature_cols].iloc[i-1].values
                
#                 # 타겟 (다음 7일)
#                 target = group['매출수량'].iloc[i:i+7].values
                
#                 # Hurdle targets
#                 zero_target = (target == 0).astype(float)  # 0인지 여부
#                 positive_target = np.where(target > 0, target, 1)  # 양수인 경우의 값
                
#                 sequences.append(seq.reshape(-1, 1))
#                 features.append(feat)
#                 zero_targets.append(zero_target)
#                 positive_targets.append(positive_target)
        
#         return np.array(sequences), np.array(features), np.array(zero_targets), np.array(positive_targets)
    
#     def train_menu_model(self, menu_data, menu_name):
#         """개별 메뉴 Hurdle Model 학습"""
#         if len(menu_data) < self.lookback_window + 14:
#             # 데이터 부족 시 단순 모델
#             mean_value = menu_data['매출수량'].mean()
#             zero_rate = (menu_data['매출수량'] == 0).mean()
#             return {
#                 'type': 'simple',
#                 'mean_value': mean_value,
#                 'zero_rate': zero_rate,
#                 'restaurant_weight': menu_data['restaurant_weight'].iloc[0]
#             }
        
#         try:
#             # Hurdle 데이터 준비
#             sequences, features, zero_targets, positive_targets = self.prepare_hurdle_data(menu_data)
            
#             if len(sequences) < 10:
#                 mean_value = menu_data['매출수량'].mean()
#                 zero_rate = (menu_data['매출수량'] == 0).mean()
#                 return {
#                     'type': 'simple',
#                     'mean_value': mean_value,
#                     'zero_rate': zero_rate,
#                     'restaurant_weight': menu_data['restaurant_weight'].iloc[0]
#                 }
            
#             # 데이터셋 및 DataLoader
#             dataset = HurdleDataset(sequences, features, zero_targets, positive_targets)
#             dataloader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)
            
#             # 모델 생성 (하이퍼파라미터 튜닝)
#             model = NHiTSHurdleModel(
#                 input_size=1,
#                 feature_size=len(self.feature_cols),
#                 hidden_size=64,  # 32 → 64로 증가
#                 num_stacks=2     # 3 → 2로 감소 (과적합 방지)
#             )
            
#             # 손실 함수 (가중치 조정)
#             bce_loss = nn.BCELoss()
#             mse_loss = nn.MSELoss()
            
#             # 옵티마이저 (학습률 조정)
#             optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)  # lr 감소, weight_decay 추가
            
#             # 학습 (에포크 증가)
#             model.train()
#             for epoch in range(100):  # 50 → 100으로 증가
#                 total_loss = 0
#                 for batch_seq, batch_feat, batch_zero, batch_pos in dataloader:
#                     optimizer.zero_grad()
                    
#                     zero_prob, positive_values, forecast = model(batch_seq, batch_feat)
                    
#                     # Hurdle Loss (가중치 조정)
#                     zero_loss = bce_loss(zero_prob, batch_zero)
#                     positive_loss = mse_loss(positive_values, batch_pos)
                    
#                     # Combined loss (가중치 실험)
#                     loss = 0.3 * zero_loss + 0.7 * positive_loss  # zero:positive = 3:7
                    
#                     loss.backward()
#                     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping 추가
#                     optimizer.step()
#                     total_loss += loss.item()
                
#                 if epoch % 30 == 0:  # 더 적게 출력
#                     avg_loss = total_loss / len(dataloader)
#                     # print(f"    {menu_name[:20]} Epoch {epoch}, Loss: {avg_loss:.4f}")
            
#             model.eval()
            
#             return {
#                 'type': 'hurdle',
#                 'model': model,
#                 'restaurant_weight': menu_data['restaurant_weight'].iloc[0]
#             }
            
#         except Exception as e:
#             # 오류 시 단순 모델
#             mean_value = menu_data['매출수량'].mean()
#             zero_rate = (menu_data['매출수량'] == 0).mean()
#             return {
#                 'type': 'simple',
#                 'mean_value': mean_value,
#                 'zero_rate': zero_rate,
#                 'restaurant_weight': menu_data['restaurant_weight'].iloc[0]
#             }
    
#     def train(self, train_df):
#         """전체 모델 학습"""
#         print("🚀 친구 스타일 N-HiTS + Hurdle 학습 시작...")
        
#         # 친구 스타일 전처리
#         train_df = self.preprocessor.create_friend_features(train_df)
#         train_df = self.preprocessor.create_lag_features(train_df, is_train=True)
#         train_df = self.preprocessor.encode_features(train_df, is_train=True)
        
#         # 피처 선택 (친구 스타일: 단순하게)
#         self.feature_cols = [
#             'month', 'dayofweek', '주중', '주말', '공휴일',
#             '요일_encoded', 'menu_category_encoded', 'restaurant_encoded',
#             'restaurant_weight', 'lag_7', 'lag_14', 'ma_7'
#         ]
        
#         # 존재하는 피처만 사용
#         self.feature_cols = [col for col in self.feature_cols if col in train_df.columns]
#         print(f"   사용 피처: {len(self.feature_cols)}개")
        
#         success_count = 0
#         fail_count = 0
        
#         # 메뉴별 Hurdle Model 학습
#         for menu_name, group in tqdm(train_df.groupby('영업장명_메뉴명'), desc="Hurdle 학습"):
#             try:
#                 model_info = self.train_menu_model(group, menu_name)
#                 self.models[menu_name] = model_info
#                 success_count += 1
#             except Exception as e:
#                 fail_count += 1
        
#         print(f"✅ 학습 완료: 성공 {success_count}개, 실패 {fail_count}개")
#         return self.models
    
#     def predict_menu(self, menu_data, menu_name):
#         """개별 메뉴 Hurdle 예측"""
#         if menu_name not in self.models:
#             return [0] * 7
        
#         model_info = self.models[menu_name]
        
#         if model_info['type'] == 'simple':
#             # 단순 모델
#             mean_val = model_info['mean_value']
#             zero_rate = model_info['zero_rate']
            
#             # Hurdle 방식으로 예측
#             predictions = []
#             for _ in range(7):
#                 if np.random.random() < zero_rate:
#                     pred = 0
#                 else:
#                     pred = max(0, np.random.poisson(mean_val))
#                 predictions.append(pred)
            
#             return predictions
        
#         try:
#             # Hurdle Model 예측
#             model = model_info['model']
            
#             if len(menu_data) >= self.lookback_window:
#                 # 최근 28일 데이터
#                 recent_data = menu_data.iloc[-self.lookback_window:].copy()
                
#                 # 입력 준비
#                 sequence = recent_data['매출수량'].values.reshape(1, -1, 1)
#                 features = recent_data[self.feature_cols].iloc[-1].values.reshape(1, -1)
                
#                 # 예측
#                 model.eval()
#                 with torch.no_grad():
#                     sequence_tensor = torch.FloatTensor(sequence)
#                     features_tensor = torch.FloatTensor(features)
                    
#                     zero_prob, positive_values, _ = model(sequence_tensor, features_tensor)
                    
#                     # Hurdle 예측 결합
#                     zero_prob_np = zero_prob[0].numpy()
#                     positive_values_np = positive_values[0].numpy()
                    
#                     predictions = []
#                     for i in range(7):
#                         if zero_prob_np[i] > 0.5:  # 0일 확률이 높으면
#                             pred = 0
#                         else:
#                             pred = positive_values_np[i]
#                         predictions.append(max(0, round(pred)))
                    
#                     return predictions
#             else:
#                 # 데이터 부족
#                 avg_value = menu_data['매출수량'].mean()
#                 return [max(0, round(avg_value)) for _ in range(7)]
                
#         except Exception as e:
#             # 오류 시 평균값
#             avg_value = menu_data['매출수량'].mean() if len(menu_data) > 0 else 0
#             return [max(0, round(avg_value)) for _ in range(7)]
    
#     def predict(self, test_df, test_prefix):
#         """예측 수행"""
#         # 동일한 전처리
#         test_df = self.preprocessor.create_friend_features(test_df)
#         test_df = self.preprocessor.create_lag_features(test_df, is_train=False)
#         test_df = self.preprocessor.encode_features(test_df, is_train=False)
        
#         results = []
        
#         for menu_name in test_df['영업장명_메뉴명'].unique():
#             menu_data = test_df[test_df['영업장명_메뉴명'] == menu_name]
            
#             if len(menu_data) == 0:
#                 pred_values = [0] * 7
#             else:
#                 pred_values = self.predict_menu(menu_data, menu_name)
            
#             # 결과 저장
#             for day, pred_val in enumerate(pred_values, 1):
#                 results.append({
#                     '영업일자': f"{test_prefix}+{day}일",
#                     '영업장명_메뉴명': menu_name,
#                     '매출수량': pred_val
#                 })
        
#         return pd.DataFrame(results)

# def main():
#     """메인 실행 함수"""
#     print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
#     # 1. 데이터 로드
#     print("\n📖 데이터 로딩...")
#     try:
#         train_df = pd.read_csv('./train.csv')
#         sample_submission = pd.read_csv('./sample_submission.csv')
#         print(f"   Train: {len(train_df):,}개, 메뉴: {train_df['영업장명_메뉴명'].nunique()}개")
#     except FileNotFoundError as e:
#         print(f"❌ 파일 없음: {e}")
#         return
    
#     # 2. 친구 스타일 모델 학습
#     print("\n🚀 친구 스타일 N-HiTS + Hurdle 학습...")
#     model = FriendStyleNHiTS()
#     model.train(train_df)
    
#     # 3. 예측
#     print("\n🔮 예측 수행...")
#     all_predictions = []
    
#     test_files = sorted(glob.glob('./test/TEST_*.csv'))
#     if not test_files:
#         test_files = sorted(glob.glob('./TEST_*.csv'))
    
#     print(f"   테스트 파일: {len(test_files)}개")
    
#     for test_file in tqdm(test_files, desc="예측"):
#         try:
#             test_df = pd.read_csv(test_file)
#             filename = os.path.basename(test_file)
#             test_prefix = re.search(r'(TEST_\d+)', filename).group(1)
            
#             pred_df = model.predict(test_df, test_prefix)
#             all_predictions.append(pred_df)
#         except Exception as e:
#             print(f"❌ {test_file}: {e}")
#             continue
    
#     if not all_predictions:
#         print("❌ 예측 실패")
#         return
    
#     # 4. 결과 통합
#     full_predictions = pd.concat(all_predictions, ignore_index=True)
#     print(f"   예측 레코드: {len(full_predictions):,}개")
    
#     # 5. 제출 형식 변환
#     print("\n💾 제출 파일 생성...")
#     pred_dict = {}
#     for _, row in full_predictions.iterrows():
#         key = (row['영업일자'], row['영업장명_메뉴명'])
#         pred_dict[key] = row['매출수량']
    
#     submission = sample_submission.copy()
#     for idx, row in submission.iterrows():
#         date = row['영업일자']
#         for col in submission.columns[1:]:
#             key = (date, col)
#             submission.loc[idx, col] = pred_dict.get(key, 0)
    
#     # 6. 후처리
#     numeric_cols = submission.columns[1:]
#     submission[numeric_cols] = submission[numeric_cols].clip(lower=0)
#     submission[numeric_cols] = submission[numeric_cols].round().astype(int)
    
#     # 7. 저장
#     output_filename = f'friend_style_nhits_hurdle_{datetime.now().strftime("%m%d_%H%M")}.csv'
#     submission.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
#     # 8. 결과 분석
#     pred_values = submission.iloc[:, 1:].values.flatten()
#     print(f"\n📈 최종 결과")
#     print(f"   출력 파일: {output_filename}")
#     print(f"   예측값 통계:")
#     print(f"     - 총 예측: {len(pred_values):,}개")
#     print(f"     - 0인 예측: {(pred_values == 0).sum():,}개 ({(pred_values == 0).mean()*100:.1f}%)")
#     print(f"     - 양수 예측: {(pred_values > 0).sum():,}개 ({(pred_values > 0).mean()*100:.1f}%)")
#     print(f"     - 평균: {pred_values.mean():.2f}")
#     print(f"     - 최대: {pred_values.max()}")
    
#     print(f"\n🎯 기대 효과:")
#     print("   ✅ 친구의 단순하지만 효과적인 전처리")
#     print("   ✅ Hurdle Model로 0 예측 개선")
#     print("   ✅ N-HiTS 계층적 예측")
#     print("   ✅ 메뉴 카테고리 정보 활용")
#     print("   🎯 목표: 친구 N-HiTS 0.69 달성!")
    
#     print(f"\n완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# if __name__ == "__main__":
#     main()


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import json
import glob
import re
import os
from datetime import datetime, timedelta
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print("🎯 진짜 Hurdle Loss N-HiTS 모델 v2.0")
print("=" * 60)
print("친구 스타일 + 진짜 Hurdle Loss (Two-Stage 제거)")

# 친구 파일 경로
FRIEND_DATA_PATH = '/home/droppgs/lg-aimers-demand-forecasting/lstm_project/joonsik'

# 데이콘 공식 매장별 가중치
RESTAURANT_WEIGHTS = {
    '미라시아': 7.71,
    '담하': 6.51,
    '연회장': 3.48,
    '라그로타': 3.44,
    '느티나무 셀프BBQ': 2.78,
    '화담숲주막': 1.43,
    '카페테리아': 1.31,
    '화담숲카페': 1.14,
    '포레스트릿': 1.00
}

class FriendStylePreprocessor:
    """친구 스타일 전처리기"""
    
    def __init__(self):
        self.label_encoders = {}
        self.menu_categories = {}
        
    def load_friend_data(self):
        """친구의 전처리된 데이터 로드"""
        preprocess_path = os.path.join(FRIEND_DATA_PATH, 'data/final_train_datasets/preprocess_train.csv')
        
        try:
            df = pd.read_csv(preprocess_path)
            print(f"✅ 친구 전처리 데이터 로드: {len(df):,}개")
            return df
        except FileNotFoundError:
            print("❌ 친구 파일 없음 - 기본 데이터 사용")
            return None
    
    def load_menu_prices(self):
        """친구의 메뉴 가격 정보 로드"""
        price_path = os.path.join(FRIEND_DATA_PATH, 'data/preprocessing_artifacts/outer/menu_price.json')
        
        try:
            with open(price_path, 'r', encoding='utf-8') as f:
                menu_data = json.load(f)
            
            categories = {}
            
            for restaurant, menus in menu_data.items():
                for menu_info in menus:
                    menu_name = menu_info['name']
                    full_name = f"{restaurant}_{menu_name}"
                    
                    # 카테고리 정보
                    if 'category' in menu_info:
                        categories[full_name] = menu_info['category']
                    else:
                        # 메뉴명으로 카테고리 추론
                        if any(drink in menu_name.lower() for drink in ['스프라이트', '콜라', '아이스티', '에이드']):
                            categories[full_name] = '음료'
                        elif any(alcohol in menu_name.lower() for alcohol in ['맥주', '소주', '막걸리', '와인', '하이볼', '칵테일']):
                            categories[full_name] = '주류'
                        elif any(food in menu_name.lower() for food in ['커피', '아메리카노', '라떼']):
                            categories[full_name] = '커피'
                        else:
                            categories[full_name] = '음식'
            
            self.menu_categories = categories
            print(f"✅ 메뉴 카테고리 정보: {len(categories)}개")
            return categories
            
        except FileNotFoundError:
            print("❌ 메뉴 가격 파일 없음")
            return {}
    
    def create_friend_features(self, df):
        """친구 스타일 피처 생성"""
        df = df.copy()
        
        # 친구 전처리 데이터가 있으면 사용
        friend_df = self.load_friend_data()
        if friend_df is not None:
            friend_features = ['요일', '주중', '주말', '공휴일']
            df = df.merge(
                friend_df[['영업일자', '영업장명_메뉴명'] + friend_features],
                on=['영업일자', '영업장명_메뉴명'],
                how='left'
            )
        else:
            # 기본 피처 생성
            df['영업일자'] = pd.to_datetime(df['영업일자'])
            df['요일'] = df['영업일자'].dt.day_name()
            df['주중'] = (df['영업일자'].dt.weekday < 5).astype(int)
            df['주말'] = (df['영업일자'].dt.weekday >= 5).astype(int)
            df['공휴일'] = 0
        
        # 메뉴 카테고리 정보 추가
        categories = self.load_menu_prices()
        df['menu_category'] = df['영업장명_메뉴명'].map(categories).fillna('음식')
        
        # 영업장명과 메뉴명 분리
        df['restaurant'] = df['영업장명_메뉴명'].str.split('_').str[0]
        df['menu'] = df['영업장명_메뉴명'].str.split('_', n=1).str[1]
        
        # 매장별 가중치
        df['restaurant_weight'] = df['restaurant'].map(RESTAURANT_WEIGHTS).fillna(1.0)
        
        # 기본 시간 피처
        df['영업일자'] = pd.to_datetime(df['영업일자'])
        df['month'] = df['영업일자'].dt.month
        df['day'] = df['영업일자'].dt.day
        df['dayofweek'] = df['영업일자'].dt.dayofweek
        
        return df
    
    def create_lag_features(self, df, is_train=True):
        """친구 스타일 지연 피처"""
        df = df.copy().sort_values(['영업장명_메뉴명', '영업일자'])
        
        # 간단한 지연 피처 (7, 14일만)
        for lag in [7, 14]:
            df[f'lag_{lag}'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(lag)
        
        # 간단한 이동평균 (7일만)
        df['ma_7'] = df.groupby('영업장명_메뉴명')['매출수량'].shift(1).rolling(
            window=7, min_periods=1
        ).mean()
        
        # 결측값을 0으로 처리
        lag_cols = [col for col in df.columns if col.startswith(('lag_', 'ma_'))]
        for col in lag_cols:
            df[col] = df[col].fillna(0)
        
        return df
    
    def encode_features(self, df, is_train=True):
        """피처 인코딩"""
        categorical_features = ['요일', 'menu_category', 'restaurant', 'menu']
        
        for col in categorical_features:
            if col in df.columns:
                if is_train:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    if col in self.label_encoders:
                        known_categories = set(self.label_encoders[col].classes_)
                        df[f'{col}_mapped'] = df[col].astype(str).apply(
                            lambda x: x if x in known_categories else self.label_encoders[col].classes_[0]
                        )
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[f'{col}_mapped'])
                    else:
                        df[f'{col}_encoded'] = 0
        
        return df

class SimpleDataset(Dataset):
    """단순한 데이터셋 (Two-Stage 제거)"""
    
    def __init__(self, sequences, features, targets):
        self.sequences = torch.FloatTensor(sequences)
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.features[idx], self.targets[idx]

class HurdleLoss(nn.Module):
    """진짜 Hurdle Loss 함수"""
    
    def __init__(self, zero_weight=1.0, positive_weight=1.0):
        super(HurdleLoss, self).__init__()
        self.zero_weight = zero_weight
        self.positive_weight = positive_weight
        
    def forward(self, pred, target):
        """
        Hurdle Loss: 0 예측과 양수 예측을 다르게 처리
        """
        # 0인 경우와 양수인 경우 분리
        zero_mask = (target == 0)
        positive_mask = (target > 0)
        
        total_loss = 0
        count = 0
        
        # 0 예측 손실 (0에 가깝게 예측했는지)
        if zero_mask.sum() > 0:
            zero_loss = torch.mean((pred[zero_mask] - 0) ** 2)
            total_loss += self.zero_weight * zero_loss
            count += 1
        
        # 양수 예측 손실 (실제 값에 가깝게 예측했는지)
        if positive_mask.sum() > 0:
            positive_loss = torch.mean((pred[positive_mask] - target[positive_mask]) ** 2)
            total_loss += self.positive_weight * positive_loss
            count += 1
        
        return total_loss / max(count, 1)

class NHiTSModel(nn.Module):
    """순수 N-HiTS 모델 (Hurdle은 Loss에서 처리)"""
    #hidden_size=64 -> 32, num_stacks=2->1
    
    def __init__(self, input_size, feature_size, hidden_size=32, num_stacks=1):
        super(NHiTSModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_stacks = num_stacks
        
        # Feature embedding
        self.feature_fc = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # N-HiTS Stacks
        self.stacks = nn.ModuleList()
        for i in range(num_stacks):
            stack = nn.ModuleDict({
                'linear1': nn.Linear(28 + hidden_size, hidden_size),
                'linear2': nn.Linear(hidden_size, hidden_size),
                'theta_f': nn.Linear(hidden_size, 7),  # Forecast
                'theta_b': nn.Linear(hidden_size, 28)  # Backcast
            })
            self.stacks.append(stack)
        
        # Final output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, 7),
            nn.ReLU()  # 음수 방지
        )
    
    def forward(self, sequences, features):
        batch_size = sequences.size(0)
        
        # Feature embedding
        feature_emb = self.feature_fc(features)
        
        # N-HiTS processing
        residual = sequences.squeeze(-1)  # (batch, 28)
        
        for stack in self.stacks:
            # Combine sequence and features
            combined = torch.cat([residual, feature_emb], dim=1)
            
            # Stack processing
            h1 = torch.relu(stack['linear1'](combined))
            h2 = torch.relu(stack['linear2'](h1))
            
            # Forecast and backcast
            forecast = stack['theta_f'](h2)
            backcast = stack['theta_b'](h2)
            
            # Update residual
            residual = residual - backcast
        
        # Final output
        output = self.output_layer(h2)
        
        return output

class TrueHurdleNHiTS:
    """진짜 Hurdle Loss N-HiTS 예측기"""
    
    def __init__(self, lookback_window=28):
        self.lookback_window = lookback_window
        self.preprocessor = FriendStylePreprocessor()
        self.models = {}
        self.feature_cols = []
        
    def prepare_data(self, df):
        """단순한 데이터 준비"""
        sequences = []
        features = []
        targets = []
        
        for menu_name, group in df.groupby('영업장명_메뉴명'):
            group = group.sort_values('영업일자').reset_index(drop=True)
            
            if len(group) < self.lookback_window + 7:
                continue
            
            for i in range(self.lookback_window, len(group) - 6):
                # 입력 시퀀스 (28일)
                seq = group['매출수량'].iloc[i-self.lookback_window:i].values
                
                # 특성 (마지막 날짜)
                feat = group[self.feature_cols].iloc[i-1].values
                
                # 타겟 (다음 7일) - 원본 그대로
                target = group['매출수량'].iloc[i:i+7].values
                
                sequences.append(seq.reshape(-1, 1))
                features.append(feat)
                targets.append(target)
        
        return np.array(sequences), np.array(features), np.array(targets)
    
    def train_menu_model(self, menu_data, menu_name):
        """개별 메뉴 모델 학습"""
        if len(menu_data) < self.lookback_window + 14:
            # 데이터 부족 시 단순 모델
            mean_value = menu_data['매출수량'].mean()
            zero_rate = (menu_data['매출수량'] == 0).mean()
            return {
                'type': 'simple',
                'mean_value': mean_value,
                'zero_rate': zero_rate,
                'restaurant_weight': menu_data['restaurant_weight'].iloc[0]
            }
        
        try:
            # 데이터 준비
            sequences, features, targets = self.prepare_data(menu_data)
            
            if len(sequences) < 10:
                mean_value = menu_data['매출수량'].mean()
                zero_rate = (menu_data['매출수량'] == 0).mean()
                return {
                    'type': 'simple',
                    'mean_value': mean_value,
                    'zero_rate': zero_rate,
                    'restaurant_weight': menu_data['restaurant_weight'].iloc[0]
                }
            
            # 데이터셋 및 DataLoader
            dataset = SimpleDataset(sequences, features, targets)
            dataloader = DataLoader(dataset, batch_size=min(16, len(dataset)), shuffle=True)
            
            # 모델 생성
            model = NHiTSModel(
                input_size=1,
                feature_size=len(self.feature_cols),
                hidden_size=64,
                num_stacks=2
            )
            
            # Hurdle Loss (0과 양수를 다르게 처리)
            hurdle_loss = HurdleLoss(zero_weight=1.5, positive_weight=1.0)  # 0 예측에 더 가중치
            
            # 옵티마이저
            # lr = 0.0005 -> 0.001
            # weight_decay = 1e-5 -> 0 
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            
            # 학습
            #epochs: 80 → 50
            model.train()
            for epoch in range(50):
                total_loss = 0
                for batch_seq, batch_feat, batch_target in dataloader:
                    optimizer.zero_grad()
                    
                    output = model(batch_seq, batch_feat)
                    loss = hurdle_loss(output, batch_target)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
                
                if epoch % 40 == 0:
                    avg_loss = total_loss / len(dataloader)
                    # print(f"    {menu_name[:20]} Epoch {epoch}, Hurdle Loss: {avg_loss:.4f}")
            
            model.eval()
            
            return {
                'type': 'nhits',
                'model': model,
                'restaurant_weight': menu_data['restaurant_weight'].iloc[0]
            }
            
        except Exception as e:
            # 오류 시 단순 모델
            mean_value = menu_data['매출수량'].mean()
            zero_rate = (menu_data['매출수량'] == 0).mean()
            return {
                'type': 'simple',
                'mean_value': mean_value,
                'zero_rate': zero_rate,
                'restaurant_weight': menu_data['restaurant_weight'].iloc[0]
            }
    
    def train(self, train_df):
        """전체 모델 학습"""
        print("🚀 진짜 Hurdle Loss N-HiTS 학습 시작...")
        
        # 친구 스타일 전처리
        train_df = self.preprocessor.create_friend_features(train_df)
        train_df = self.preprocessor.create_lag_features(train_df, is_train=True)
        train_df = self.preprocessor.encode_features(train_df, is_train=True)
        
        # 피처 선택 (친구 스타일)
        self.feature_cols = [
            'month', 'dayofweek', '주중', '주말', '공휴일',
            '요일_encoded', 'menu_category_encoded', 'restaurant_encoded',
            'restaurant_weight', 'lag_7', 'lag_14', 'ma_7'
        ]
        
        # 존재하는 피처만 사용
        self.feature_cols = [col for col in self.feature_cols if col in train_df.columns]
        print(f"   사용 피처: {len(self.feature_cols)}개")
        
        success_count = 0
        fail_count = 0
        
        # 메뉴별 모델 학습
        for menu_name, group in tqdm(train_df.groupby('영업장명_메뉴명'), desc="진짜 Hurdle 학습"):
            try:
                model_info = self.train_menu_model(group, menu_name)
                self.models[menu_name] = model_info
                success_count += 1
            except Exception as e:
                fail_count += 1
        
        print(f"✅ 학습 완료: 성공 {success_count}개, 실패 {fail_count}개")
        return self.models
    
    def predict_menu(self, menu_data, menu_name):
        """개별 메뉴 예측"""
        if menu_name not in self.models:
            return [0] * 7
        
        model_info = self.models[menu_name]
        
        if model_info['type'] == 'simple':
            # 단순 모델
            mean_val = model_info['mean_value']
            zero_rate = model_info['zero_rate']
            
            predictions = []
            for _ in range(7):
                if np.random.random() < zero_rate:
                    pred = 0
                else:
                    pred = max(0, np.random.poisson(mean_val))
                predictions.append(pred)
            
            return predictions
        
        try:
            # N-HiTS 예측
            model = model_info['model']
            
            if len(menu_data) >= self.lookback_window:
                # 최근 28일 데이터
                recent_data = menu_data.iloc[-self.lookback_window:].copy()
                
                # 입력 준비
                sequence = recent_data['매출수량'].values.reshape(1, -1, 1)
                features = recent_data[self.feature_cols].iloc[-1].values.reshape(1, -1)
                
                # 예측
                model.eval()
                with torch.no_grad():
                    sequence_tensor = torch.FloatTensor(sequence)
                    features_tensor = torch.FloatTensor(features)
                    
                    output = model(sequence_tensor, features_tensor)
                    predictions = output[0].numpy()
                    
                    # 후처리
                    result = [max(0, round(pred)) for pred in predictions]
                    return result
            else:
                # 데이터 부족
                avg_value = menu_data['매출수량'].mean()
                return [max(0, round(avg_value)) for _ in range(7)]
                
        except Exception as e:
            # 오류 시 평균값
            avg_value = menu_data['매출수량'].mean() if len(menu_data) > 0 else 0
            return [max(0, round(avg_value)) for _ in range(7)]
    
    def predict(self, test_df, test_prefix):
        """예측 수행"""
        # 동일한 전처리
        test_df = self.preprocessor.create_friend_features(test_df)
        test_df = self.preprocessor.create_lag_features(test_df, is_train=False)
        test_df = self.preprocessor.encode_features(test_df, is_train=False)
        
        results = []
        
        for menu_name in test_df['영업장명_메뉴명'].unique():
            menu_data = test_df[test_df['영업장명_메뉴명'] == menu_name]
            
            if len(menu_data) == 0:
                pred_values = [0] * 7
            else:
                pred_values = self.predict_menu(menu_data, menu_name)
            
            # 결과 저장
            for day, pred_val in enumerate(pred_values, 1):
                results.append({
                    '영업일자': f"{test_prefix}+{day}일",
                    '영업장명_메뉴명': menu_name,
                    '매출수량': pred_val
                })
        
        return pd.DataFrame(results)

def main():
    """메인 실행 함수"""
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. 데이터 로드
    print("\n📖 데이터 로딩...")
    try:
        train_df = pd.read_csv('./train.csv')
        sample_submission = pd.read_csv('./sample_submission.csv')
        print(f"   Train: {len(train_df):,}개, 메뉴: {train_df['영업장명_메뉴명'].nunique()}개")
    except FileNotFoundError as e:
        print(f"❌ 파일 없음: {e}")
        return
    
    # 2. 진짜 Hurdle Loss 모델 학습
    print("\n🚀 진짜 Hurdle Loss N-HiTS 학습...")
    model = TrueHurdleNHiTS()
    model.train(train_df)
    
    # 3. 예측
    print("\n🔮 예측 수행...")
    all_predictions = []
    
    test_files = sorted(glob.glob('./test/TEST_*.csv'))
    if not test_files:
        test_files = sorted(glob.glob('./TEST_*.csv'))
    
    print(f"   테스트 파일: {len(test_files)}개")
    
    for test_file in tqdm(test_files, desc="예측"):
        try:
            test_df = pd.read_csv(test_file)
            filename = os.path.basename(test_file)
            test_prefix = re.search(r'(TEST_\d+)', filename).group(1)
            
            pred_df = model.predict(test_df, test_prefix)
            all_predictions.append(pred_df)
        except Exception as e:
            print(f"❌ {test_file}: {e}")
            continue
    
    if not all_predictions:
        print("❌ 예측 실패")
        return
    
    # 4. 결과 통합
    full_predictions = pd.concat(all_predictions, ignore_index=True)
    print(f"   예측 레코드: {len(full_predictions):,}개")
    
    # 5. 제출 형식 변환
    print("\n💾 제출 파일 생성...")
    pred_dict = {}
    for _, row in full_predictions.iterrows():
        key = (row['영업일자'], row['영업장명_메뉴명'])
        pred_dict[key] = row['매출수량']
    
    submission = sample_submission.copy()
    for idx, row in submission.iterrows():
        date = row['영업일자']
        for col in submission.columns[1:]:
            key = (date, col)
            submission.loc[idx, col] = pred_dict.get(key, 0)
    
    # 6. 후처리
    numeric_cols = submission.columns[1:]
    submission[numeric_cols] = submission[numeric_cols].clip(lower=0)
    submission[numeric_cols] = submission[numeric_cols].round().astype(int)
    
    # 7. 저장
    output_filename = f'true_hurdle_nhits_{datetime.now().strftime("%m%d_%H%M")}.csv'
    submission.to_csv(output_filename, index=False, encoding='utf-8-sig')
    
    # 8. 결과 분석
    pred_values = submission.iloc[:, 1:].values.flatten()
    print(f"\n📈 최종 결과")
    print(f"   출력 파일: {output_filename}")
    print(f"   예측값 통계:")
    print(f"     - 총 예측: {len(pred_values):,}개")
    print(f"     - 0인 예측: {(pred_values == 0).sum():,}개 ({(pred_values == 0).mean()*100:.1f}%)")
    print(f"     - 양수 예측: {(pred_values > 0).sum():,}개 ({(pred_values > 0).mean()*100:.1f}%)")
    print(f"     - 평균: {pred_values.mean():.2f}")
    print(f"     - 최대: {pred_values.max()}")
    
    print(f"\n🎯 개선사항:")
    print("   ✅ Two-Stage 제거 → 단순하고 안정적")
    print("   ✅ 진짜 Hurdle Loss → 0과 양수 구분 처리")
    print("   ✅ 친구 전처리 유지 → 검증된 방식")
    print("   🎯 목표: 0.77 → 0.69 달성!")
    
    print(f"\n완료 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()