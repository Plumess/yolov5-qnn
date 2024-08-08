# Create by Hengyu & Zilong on 2024/05/15

# 获取当前用户名
CURRENT_USER=$(whoami)
echo "当前用户: $CURRENT_USER"
# 请求sudo权限以确保脚本有足够权限运行
sudo echo "获得sudo权限..."
if sudo grep -q "^$CURRENT_USER ALL=(ALL) NOPASSWD: ALL" /etc/sudoers; then
    echo "免密码sudo权限已设置。"
else
    echo "未在 sudoers 文件中找到免密码规则，正在添加..."
    # 使用sudo bash -c 执行一个命令字符串，确保所有操作都在提升权限的环境下执行
    sudo bash -c "echo '$CURRENT_USER ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers"
    # 确认更改是否成功
    if sudo grep -q "^$CURRENT_USER ALL=(ALL) NOPASSWD: ALL" /etc/sudoers; then
        echo "免密码规则已成功添加。"
    else
        echo "尝试添加免密码规则失败，请手动检查 /etc/sudoers 文件。"
    fi
fi

# 安装 Anaconda,运行下面命令，安装提示默认安装
bash Anaconda3-2024.02-1-Linux-x86_64.sh -b -p $HOME/anaconda3
# 输入下面的命令将 conda 写入 bash
eval "$(~/anaconda3/bin/conda shell.bash hook)"
conda init
conda config --set auto_activate_base false
source ~/.bashrc

# 安装CMake(Ubuntu/debian)
sudo apt-get update -y
sudo apt-get install cmake -y

## INSTALL AI ENGINE && QPM----------------------------------------------
# QNN AI Engine & QPM Version (check download .qik & .deb file)
export QIK_VERSION="2.19.4.240226"
export QIK_PATH="./qualcomm_ai_engine_direct.$QIK_VERSION.Linux-AnyCPU.qik"
export QPM_VERSION="3.0.103.0"
export QPM_PATH="./QualcommPackageManager3.$QPM_VERSION.Linux-x86.deb"

# 安装 QPM3 .deb
echo "正在安装 QPM3 ..."
sudo dpkg -i $QPM_PATH

# 需要登录 Qualcomm 账号，且该账号签写了 agreements
# 改为自己注册的 Qualcomm 账号
export USERNAME="example@mail.com"
export PASSWORD="passssword"

sudo apt-get install expect -y
# 自动登录 qpm-cli
echo "正在自动输入 Qualcomm 账号和密码 ..."
expect -c "
log_user 0;  # 关闭输出，不显示expect命令细节
spawn qpm-cli --login $USERNAME
expect \"Password:\"
send \"$PASSWORD\r\"
log_user 1;  # 打开输出，显示qpm-cli的输出
interact
log_user 0;  # 交互完成后再次关闭输出
"

# 激活许可并自动确认
echo "正在激活许可 ..."
qpm-cli --license-activate qualcomm_ai_engine_direct

# 提取 QIK 文件并自动确认
echo "正在提取 QIK 文件 ..."
expect -c "
log_user 0;
spawn qpm-cli --extract $QIK_PATH
log_user 1;
set timeout 30
expect {
    -re {Accept and continue.*\[y/n\] :} {
        send \"y\r\"
        exp_continue
    }
    timeout {
        puts \"Timeout occurred\"
        exit 1
    }
    eof
}
log_user 0;
"

# 若无${QNN_SDK_ROOT}，默认安装在 opt/qcom/aistack/qnn/$QIK_VERSION
# 可以查看 qmp3 安装 log 最后的 [info] 确认
echo "正在导入 QNN_SDK_ROOT 环境变量 ..."
export QNN_SDK_ROOT=/opt/qcom/aistack/qnn/$QIK_VERSION
echo "export QNN_SDK_ROOT=/opt/qcom/aistack/qnn/$QIK_VERSION" >> ~/.bashrc
sed -i 's/^\(\[ -z "\$PS1" \] && return\)/#\1/' ~/.bashrc

## QNN PYTHON ENVIRONMENT -----------------------------------------
# 设置 conda 路径
echo "正在配置 conda 环境，name = qnn ..."
eval "$(conda shell.bash hook)"
# 检查conda环境是否存在
env_exists=$(conda info --envs | grep '^qnn\s')
# 如果环境不存在，创建环境
if [ -z "$env_exists" ]; then
    echo "环境 'qnn' 不存在，正在创建..."
    conda create -n qnn python=3.8 -y
else
    echo "环境 'qnn' 已存在，跳过创建。"
fi
conda activate qnn

# 安装 onnx 所需环境
echo "正在安装 onnx ..."
pip install onnx -i https://mirrors.aliyun.com/pypi/simple/
pip install onnxruntime -i https://mirrors.aliyun.com/pypi/simple/
pip install onnxsim -i https://mirrors.aliyun.com/pypi/simple/
# 安装 pytorch 所需环境
echo "正在安装 pytorch ..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
# 安装 yolo 所需环境
pip install timm -i https://mirrors.aliyun.com/pypi/simple/
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

## ENV VAR --------------------------------------------------------
# 当前终端单次使用
echo "单次激活 QNN_SDK_ROOT环境 ..."
source ${QNN_SDK_ROOT}/bin/envsetup.sh
# 永久化(不推荐)
# echo 'source ${QNN_SDK_ROOT}/bin/envsetup.sh' >> ~/.bashrc

# 调用QNN_SDK_ROOT自带sh，搭建python所需环境
# 设置pip临时源并安装依赖
PIP_SOURCE="https://mirrors.aliyun.com/pypi/simple/"
export PIP_INDEX_URL=$PIP_SOURCE
export PIP_TRUSTED_HOST=$(echo $PIP_SOURCE | awk -F/ '{print $3}')
echo "正在配置 QNN SDK Python 环境 ..."
expect -c "
log_user 0;
spawn ${QNN_SDK_ROOT}/bin/check-python-dependency
log_user 1;
set timeout 120
expect {
    -re {Press \\\[ENTER\\\].*} {
        send \"\r\"
        exp_continue
    }
    timeout {
        puts \"Timeout occurred\"
        exit 1
    }
    eof {
        puts \"End of file reached\"
        exit 0
    }
}
log_user 0;
"

## LINUX ENVIRONMENT ----------------------------------------------
echo "正在配置 QNN SDK Linux 环境 ..."
# 安装 LLVM 和 Clang
sudo apt-get install -y llvm clang
# 安装 libc++ 开发库
sudo apt-get install -y libc++-dev libc++abi-dev
sudo apt-get install -y libflatbuffers-dev
sudo apt-get install -y rename
# 检查安装的版本
clang --version
hash -r

expect -c "
log_user 0;
spawn sudo ${QNN_SDK_ROOT}/bin/check-linux-dependency.sh
log_user 1;
set timeout 120
expect {
    -re {Press \\\[ENTER\\\].*} {
        send \"\r\"
        exp_continue
    }
    timeout {
        puts \"Timeout occurred\"
        exit 1
    }
    eof {
        puts \"End of file reached\"
        exit 0
    }
}
log_user 0;
"

## (Optional) ANDROID NDK  ---------------------------------------------------
# echo "(Optional) 正在配置 Android NDK 环境 ..."
# export CURRENT_DIR=$(pwd)  # /path/to/qnn/
# export NDK_PATH=$CURRENT_DIR/android-ndk-r26d
# echo "export ANDROID_NDK_HOME=$NDK_PATH" >> ~/.bashrc
# echo 'export PATH=$PATH:$ANDROID_NDK_HOME' >> ~/.bashrc

# source ~/.bashrc

echo '环境配置成功'