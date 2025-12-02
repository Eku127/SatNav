# SatNav 设计总结：卫星地图仿真器

## 设计评价

### ✅ 合理性评价：**非常合理**

您的设计将VLN任务扩展到卫星地图导航领域，这是一个有意义的应用场景。设计简洁且符合实际需求。

## 核心设计特点

### 1. 状态表示

**位置**：
- 使用地理坐标系统（WGS84）
- xyz分别表示：经度（longitude）、纬度（latitude）、高度（altitude）
- 符合地理信息系统的标准

**旋转**：
- 使用roll角度（航向角/heading）
- 范围：0-360度，0表示正北方向
- 相比quaternion更简洁，适合地面导航

### 2. 图像生成

**方式**：从现有卫星地图crop图像
- ✅ 高效：避免复杂的3D渲染
- ✅ 真实：使用真实的卫星地图数据
- ✅ 灵活：可以支持不同分辨率和缩放级别

**实现要点**：
- 根据智能体的位置（经纬度）确定crop中心
- 根据roll角度确定crop方向
- 根据配置的crop_size确定crop区域大小

### 3. 距离计算

**使用测地距离**（geodesic distance）：
- 考虑地球曲率
- 使用Haversine或Vincenty公式
- 不是简单的欧氏距离

## 已修正的文档内容

### 1. AgentState定义
- 位置：从 `[x, y, z]` 改为 `[longitude, latitude, altitude]`
- 旋转：从 `quaternion [x, y, z, w]` 改为 `roll角度（float）`

### 2. Episode数据结构
- `start_position`: 明确为经纬度+高度
- `start_rotation`: 从quaternion改为roll角度
- `reference_path`: 路径点使用经纬度+高度

### 3. 传感器说明
- RGB传感器：明确从卫星地图crop
- 已移除Depth传感器，仅使用RGB图像

### 4. 距离计算
- 所有距离计算明确使用测地距离
- 强调考虑地球曲率
- PathLength也使用测地距离累加

### 5. 仿真器接口
- `get_agent_state()`: 返回经纬度+高度和roll角度
- `set_agent_state()`: 参数说明更新
- `get_observations()`: 说明从卫星地图crop
- `geodesic_distance()`: 强调使用Haversine/Vincenty公式

## 需要注意的实现细节

### 1. 坐标转换
```python
# 经纬度到xyz的转换（如果需要）
def lon_lat_alt_to_xyz(lon, lat, alt):
    # 考虑地球曲率
    # 使用合适的投影系统（如UTM）
    pass
```

### 2. 测地距离计算
```python
def geodesic_distance(lon1, lat1, lon2, lat2):
    # 使用Haversine公式
    # 或更精确的Vincenty公式
    pass
```

### 3. 图像裁剪
```python
def crop_satellite_image(lon, lat, roll, crop_size_meters):
    # 1. 根据经纬度确定crop中心
    # 2. 根据roll角度确定方向
    # 3. 根据crop_size_meters确定像素范围
    # 4. 从卫星地图数据中crop
    pass
```

### 4. 动作执行
```python
# MOVE_FORWARD需要考虑地球曲率
def move_forward(lon, lat, roll, step_size_meters):
    # 根据roll角度和step_size计算新的经纬度
    # 考虑地球曲率
    pass
```

## 与室内导航的对比

| 特性 | 室内导航（Habitat） | 卫星地图导航（SatNav） |
|------|-------------------|---------------------|
| **场景** | 3D室内场景 | 卫星地图 |
| **位置** | x, y, z (米) | 经度, 纬度, 高度 |
| **旋转** | Quaternion (4D) | Roll角度 (1D) |
| **坐标系统** | 局部3D坐标 | 地理坐标（WGS84） |
| **距离** | 欧氏距离 | 测地距离（Haversine） |
| **图像** | 3D渲染 | 卫星地图裁剪 |
| **深度** | 有（3D场景） | 无或高度图 |
| **范围** | 单个建筑物 | 整个地球表面 |

## 总结

您的设计**非常合理**，主要优势：

1. ✅ **应用场景真实**：卫星地图导航是实际应用
2. ✅ **状态表示简洁**：经纬度+roll角度，符合地理信息系统标准
3. ✅ **实现方式高效**：从卫星地图crop，避免复杂渲染
4. ✅ **设计简洁**：相比quaternion，roll角度更简单

主要需要注意：
1. ⚠️ 坐标转换和距离计算的准确性（考虑地球曲率）
2. ⚠️ 图像裁剪参数的合理设置
3. ⚠️ 深度信息的处理（移除或使用高度图）

所有文档已根据您的描述进行了修正，现在准确反映了卫星地图仿真器的特点。

