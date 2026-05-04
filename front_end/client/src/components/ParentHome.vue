<template>
  <div class="parent-home">
    <!-- 顶部问候 -->
    <div class="welcome-card">
      <div class="welcome-text">
        <h2>👋 家长您好</h2>
        <p>{{ dateText }}</p>
      </div>
      <div class="avatar">👨‍👩</div>
    </div>

    <!-- 孩子信息卡片 -->
    <div class="kid-card" v-if="kidInfo">
      <div class="kid-header">
        <div class="label">监护孩子</div>
        <div class="kid-name">{{ kidInfo.student_name }}</div>
      </div>
      <div class="kid-info">
        <div>
          <span>班级</span>
          <strong>{{ kidInfo.class_name }}</strong>
        </div>
        <div>
          <span>今日专注</span>
          <strong class="purple">{{ kidInfo.today_focus }}%</strong>
        </div>
        <div>
          <span>总报告</span>
          <strong>{{ kidInfo.total_reports }} 份</strong>
        </div>
      </div>
    </div>

    <!-- 快捷功能（只保留你有的页面） -->
    <div class="grid-group">
      <div class="grid-item" @click="$router.push('/report')">
        <div class="icon">📊</div>
        <div class="title">学情报告</div>
        <div class="desc">查看课堂表现记录</div>
      </div>

      <div class="grid-item" @click="$router.push('/suggest')">
        <div class="icon">💡</div>
        <div class="title">共育建议</div>
        <div class="desc">AI + 老师指导方案</div>
      </div>
    </div>

    <!-- 温馨提示 -->
    <div class="section-card">
      <h3>💡 家长监护说明</h3>
      <div class="tips">
        <div class="tip-item">
          <span>•</span>
          <p>实时查看孩子每节课的课堂行为报告</p>
        </div>
        <div class="tip-item">
          <span>•</span>
          <p>系统自动生成专注度、坐姿、举手等统计数据</p>
        </div>
        <div class="tip-item">
          <span>•</span>
          <p>根据课堂表现提供专业的家庭共育建议</p>
        </div>
      </div>
    </div>

    <!-- 底部 -->
    <div class="footer">智慧课堂家长端 © 2025 毕业设计</div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const kidInfo = ref(null)
const dateText = ref('')

// 自动获取今天日期
function getToday() {
  const now = new Date()
  const y = now.getFullYear()
  const m = String(now.getMonth() + 1).padStart(2, '0')
  const d = String(now.getDate()).padStart(2, '0')
  const w = ['日', '一', '二', '三', '四', '五', '六'][now.getDay()]
  return `${y}年${m}月${d}日 星期${w}`
}

// 加载家长首页数据
onMounted(async () => {
  dateText.value = getToday()

  const user = JSON.parse(localStorage.getItem('currentUser'))
  const user_code = user?.user_code
  if (!user_code) return

  try {
    const { data } = await axios.get('http://localhost:5002/api/parent/home', {
      params: { user_code }
    })
    kidInfo.value = data
  } catch (err) {
    console.error('加载失败', err)
  }
})
</script>

<style scoped>
.parent-home {
  padding: 20px;
  background: linear-gradient(to bottom, #f9faff, #f1f5ff);
  min-height: 100vh;
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
}

/* 问候卡片 */
.welcome-card {
  background: linear-gradient(90deg, #7c5fff, #9b77ff);
  color: white;
  border-radius: 18px;
  padding: 22px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.welcome-text h2 {
  margin: 0 0 4px 0;
  font-size: 22px;
}
.welcome-text p {
  margin: 0;
  font-size: 13px;
  opacity: 0.9;
}
.avatar {
  width: 52px;
  height: 52px;
  background: rgba(255,255,255,0.2);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
}

/* 孩子信息卡片 */
.kid-card {
  background: #fff;
  border-radius: 18px;
  padding: 20px;
  margin-bottom: 18px;
  box-shadow: 0 3px 10px rgba(0,0,0,0.05);
}
.kid-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 14px;
}
.kid-header .label {
  font-size: 14px;
  color: #888;
}
.kid-header .kid-name {
  font-weight: bold;
  font-size: 18px;
  color: #222;
}
.kid-info {
  display: flex;
  justify-content: space-between;
  text-align: center;
}
.kid-info div {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.kid-info span {
  font-size: 12px;
  color: #999;
}
.kid-info strong {
  font-size: 15px;
  font-weight: bold;
}
.purple { color: #7c5fff; }

/* 功能网格 */
.grid-group {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 14px;
  margin-bottom: 20px;
}
.grid-item {
  background: white;
  border-radius: 18px;
  padding: 22px 16px;
  box-shadow: 0 3px 8px rgba(0,0,0,0.04);
  cursor: pointer;
  transition: 0.2s;
}
.grid-item:active {
  transform: scale(0.97);
}
.grid-item .icon {
  font-size: 28px;
  margin-bottom: 10px;
}
.grid-item .title {
  font-weight: bold;
  font-size: 16px;
  margin-bottom: 4px;
}
.grid-item .desc {
  font-size: 12px;
  color: #999;
}

/* 提示卡片 */
.section-card {
  background: white;
  border-radius: 18px;
  padding: 20px;
  margin-bottom: 20px;
}
.section-card h3 {
  margin: 0 0 12px 0;
  font-size: 16px;
}
.tips {
  line-height: 1.6;
}
.tip-item {
  display: flex;
  gap: 8px;
  margin-bottom: 6px;
  font-size: 14px;
  color: #555;
}
.tip-item span {
  color: #7c5fff;
  font-weight: bold;
}

.footer {
  text-align: center;
  font-size: 12px;
  color: #ccc;
  margin-top: 10px;
}
</style>