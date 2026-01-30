# UV 测试环境管理指南

本指南介绍如何使用 **uv** (Astral 公司开发的极速 Python 包管理器) 管理 Smart Document Assistant 的测试环境。

## 为什么选择 uv？

| 对比项 | pip + venv | uv | 优势 |
|--------|-----------|-----|------|
| **安装速度** | ~30-60秒 | ⚡ **3-10秒** | 快 10-20 倍 |
| **依赖解析** | 较慢 | 极快 (Rust) | 秒级解析 |
| **全局缓存** | ❌ | ✅ | 节省磁盘空间 |
| **Lock 文件** | 手动维护 | 自动生成 | 环境可复现 |
| **兼容性** | 标准 | 100% pip 兼容 | 无缝迁移 |

## 快速开始

### 1. 安装 uv

```bash
# Linux/Mac
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 验证安装
uv --version  # 应显示版本号，如 uv 0.5.x
```

### 2. 初始化测试环境

```bash
cd /home/mac/Rav4Agent

# 使用脚本自动设置
./test.sh setup

# 或手动执行
cd backend
uv venv .venv-test
uv pip install -e ".[test]" --python .venv-test
```

### 3. 运行测试

```bash
# 使用脚本 (推荐)
./test.sh unit              # 运行单元测试
./test.sh coverage          # 运行带覆盖率
./test.sh document          # 运行文档相关测试

# 或使用 Makefile
make test                   # 快速测试
make test-coverage          # 带覆盖率
make test-document          # 文档测试

# 或使用 uv 直接运行
cd backend
uv run --python .venv-test pytest tests/unit -v
```

## 环境结构

```
Rav4Agent/
├── backend/
│   ├── .venv-test/              # 测试专用环境 (gitignored)
│   ├── pyproject.toml           # uv 主配置
│   ├── pytest.ini              # 测试配置
│   ├── requirements.txt         # 生产依赖 (兼容性)
│   ├── requirements-test.txt    # 测试依赖 (兼容性)
│   └── tests/
│       ├── conftest.py         # 共享 fixtures
│       ├── unit/
│       │   ├── test_document_parsing.py
│       │   ├── test_node_vectorization.py
│       │   ├── test_vector_storage.py
│       │   ├── test_vector_retrieval.py
│       │   └── test_prompt_generation.py
│       └── README.md
├── test.sh                      # Unix/Linux/Mac 测试脚本
├── test.bat                     # Windows 测试脚本
└── Makefile                     # 统一命令入口
```

## 详细命令参考

### 测试脚本 (test.sh / test.bat)

```bash
# Unix/Mac/Linux
./test.sh [command]

# Windows
test.bat [command]
```

| 命令 | 说明 | 示例 |
|------|------|------|
| `setup` | 初始化测试环境 | `./test.sh setup` |
| `unit` | 运行单元测试 (默认) | `./test.sh` |
| `all` | 运行所有测试 (含慢测试) | `./test.sh all` |
| `coverage` | 带覆盖率报告 | `./test.sh coverage` |
| `document` | 文档解析测试 | `./test.sh document` |
| `vector` | 向量操作测试 | `./test.sh vector` |
| `embedding` | 嵌入生成测试 | `./test.sh embedding` |
| `storage` | 存储测试 | `./test.sh storage` |
| `retrieval` | 检索测试 | `./test.sh retrieval` |
| `prompt` | 提示词测试 | `./test.sh prompt` |
| `pattern` | 按名称模式匹配 | `./test.sh pattern split` |
| `clean` | 清理环境 | `./test.sh clean` |

### Makefile

```bash
# 安装
make install          # 生产依赖
make install-test     # 测试依赖
make install-dev      # 开发依赖

# 测试
make test             # 单元测试
make test-all         # 所有测试
make test-coverage    # 覆盖率
make test-document    # 文档测试
make test-vector      # 向量测试
make test-embedding   # 嵌入测试
make test-storage     # 存储测试
make test-retrieval   # 检索测试
make test-prompt      # 提示词测试

# 代码质量
make lint             # 代码检查
make format           # 代码格式化
make type-check       # 类型检查

# 维护
make clean            # 清理缓存
make clean-all        # 完全清理
```

## pyproject.toml 详解

### 核心配置

```toml
[project]
name = "smart-document-assistant"
version = "1.0.0"
dependencies = [
    # 生产依赖
]

[project.optional-dependencies]
test = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    # ... 测试依赖
]
dev = [
    "black>=23.0.0",
    "ruff>=0.1.0",
    # ... 开发依赖
]

[tool.uv]
# uv 专用配置
test-dependencies = [
    "pytest>=7.4.0",
    # 可独立于 pyproject.toml 配置
]
```

### 依赖安装方式

```bash
# 1. 安装主依赖 + 测试依赖 (推荐)
uv pip install -e ".[test]"

# 2. 安装主依赖 + 开发依赖
uv pip install -e ".[dev]"

# 3. 安装全部
uv pip install -e ".[test,dev]"

# 4. 仅安装主依赖
uv pip install -e .
```

## 测试配置 (pytest.ini)

```ini
[pytest]
testpaths = tests
pythonpath = .
addopts = -v --tb=short --cov=app

markers =
    slow: 慢测试
    unit: 单元测试
    document: 文档测试
    vector: 向量测试
```

## CI/CD 集成

### GitHub Actions (已配置)

```yaml
# .github/workflows/tests.yml
- name: Install uv
  uses: astral-sh/setup-uv@v3

- name: Setup and test
  run: |
    uv venv .venv
    uv pip install -e ".[test]" --python .venv
    uv run --python .venv pytest tests/unit
```

**特点：**
- ✅ 无需 ChromaDB 服务 (单元测试使用 Mock)
- ✅ 支持 Python 3.8-3.11 矩阵测试
- ✅ 自动缓存 uv 依赖
- ✅ 并行运行各类别测试

## 常见问题

### Q: 需要安装 ChromaDB 吗？
**A: 不需要！** 单元测试使用 Mock 对象，无需真实服务。

```python
@patch('chromadb.HttpClient')
def test_chroma(self, mock_client):
    # Mock 测试，无需真实 Chroma
```

### Q: 测试失败怎么办？
**A:** 检查以下步骤：

```bash
# 1. 清理并重建环境
./test.sh clean
./test.sh setup

# 2. 单独运行失败测试
uv run pytest tests/unit/test_specific.py::TestClass::test_method -v

# 3. 查看详细错误
uv run pytest tests/unit --tb=long -v
```

### Q: 如何添加新测试？
**A:** 在 `backend/tests/unit/` 创建新文件：

```python
# test_new_feature.py
import pytest

class TestNewFeature:
    """测试新功能"""
    
    def test_basic_functionality(self):
        """测试基本功能"""
        assert True
    
    @pytest.mark.slow
    def test_slow_operation(self):
        """慢测试，默认不运行"""
        import time
        time.sleep(1)
        assert True
```

### Q: 覆盖率报告在哪看？
**A:** 运行 `make test-coverage` 或 `./test.sh coverage`：

```
# 终端显示摘要
# 同时生成 htmlcov/index.html
open htmlcov/index.html  # 浏览器打开详细报告
```

### Q: 支持 Windows 吗？
**A:** 完全支持！使用 `test.bat`：

```cmd
test.bat unit
test.bat coverage
test.bat document
```

## 性能对比

在实际项目中测试：

| 操作 | pip | uv | 提升 |
|------|-----|-----|------|
| 安装依赖 (首次) | 45s | 8s | **5.6x** |
| 安装依赖 (缓存) | 12s | 2s | **6x** |
| 创建 venv | 3s | 0.5s | **6x** |
| 解析依赖 | 5s | 0.2s | **25x** |

## 迁移指南

从 pip 迁移到 uv：

```bash
# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 创建 pyproject.toml (已完成)
# 包含所有依赖定义

# 3. 使用 uv 安装
uv pip install -e ".[test]"

# 4. 运行测试
uv run pytest

# 5. (可选) 生成 lock 文件
uv pip compile pyproject.toml -o requirements.lock
```

## 参考链接

- [uv 官方文档](https://docs.astral.sh/uv/)
- [pytest 文档](https://docs.pytest.org/)
- [项目测试 README](./backend/tests/README.md)

---

**提示：** 所有配置文件已就绪，只需安装 uv 即可开始使用！
