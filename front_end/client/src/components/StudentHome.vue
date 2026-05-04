<template>
  <div class="student-home">
    <!-- 顶部问候 -->
    <div class="welcome-card">
      <div class="welcome-text">
        <h2>👋 你好，{{ studentName }}</h2>
        <p>{{ dateText }} · {{ className }}</p>
      </div>
      <div class="avatar">👦</div>
    </div>

    <!-- 快捷功能入口（只保留你真实有的页面） -->
    <div class="grid-group">
      <div class="grid-item" @click="$router.push('/behavior')">
        <div class="icon">📊</div>
        <div class="title">行为报告</div>
        <div class="desc">查看课堂表现记录</div>
      </div>

      <div class="grid-item" @click="$router.push('/medal')">
        <div class="icon">🎖️</div>
        <div class="title">我的勋章</div>
        <div class="desc">荣誉成长体系</div>
      </div>
    </div>

    <!-- 使用帮助 -->
    <div class="section-card">
      <h3>📘 使用说明</h3>
      <div class="help-content">
        <div class="help-item">
          <span>1.</span>
          <p>每节课后自动生成课堂行为分析报告</p>
        </div>
        <div class="help-item">
          <span>2.</span>
          <p>专注度表现越好，获得的荣誉勋章越多</p>
        </div>
        <div class="help-item">
          <span>3.</span>
          <p>实时查看老师评价与AI学习建议</p>
        </div>
      </div>
    </div>

    <!-- 关于系统 -->
    <div class="section-card dark">
      <h3>ℹ️ 关于系统</h3>
      <div class="about-content">
        <p>智慧课堂行为分析系统 · 学生端</p>
        <p>基于AI视觉分析 | 专注度监测 | 成长激励</p>
        <p class="version">v1.0.0</p>
      </div>
    </div>

    <!-- 底部 -->
    <div class="footer">
      智慧课堂 © 2025 毕业设计专用
    </div>
  </div>
  <StudentAiFloat></StudentAiFloat>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import StudentAiFloat from '../components/StudentAiFloat.vue'

const studentName = ref('加载中...')
const className = ref('')
const dateText = ref('')

// 获取今天日期
function getTodayText() {
  const now = new Date()
  const year = now.getFullYear()
  const month = String(now.getMonth() + 1).padStart(2, '0')
  const day = String(now.getDate()).padStart(2, '0')
  const week = ['日', '一', '二', '三', '四', '五', '六'][now.getDay()]
  return `${year}年${month}月${day}日 星期${week}`
}

onMounted(async () => {
  dateText.value = getTodayText()

  const user = JSON.parse(localStorage.getItem('currentUser'))
  const student_code = user?.student_code
  if (!student_code) return

  try {
    const { data } = await axios.get('http://localhost:5002/api/student/home', {
      params: { student_code }
    })
    studentName.value = data.student_name
    className.value = data.class_name
  } catch (err) {
    console.error('加载首页失败', err)
  }
})
</script>

<style scoped>
.student-home {
  padding: 20px;
  background: linear-gradient(to bottom, #f8faff, #eff4ff);
  min-height: 100vh;
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
}

/* 问候卡片 */
.welcome-card {
  background: linear-gradient(90deg, #429dff, #57b9ff);
  color: white;
  border-radius: 18px;
  padding: 22px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
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
  background: rgba(255, 255, 255, 0.2);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
}

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
  box-shadow: 0 3px 10px rgba(0, 0, 0, 0.05);
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

/* 区块卡片 */
.section-card {
  background: white;
  border-radius: 18px;
  padding: 20px;
  margin-bottom: 16px;
}
.section-card.dark {
  background: #f0f7ff;
}
.section-card h3 {
  margin: 0 0 12px 0;
  font-size: 16px;
}

.help-content {
  line-height: 1.6;
}
.help-item {
  display: flex;
  gap: 8px;
  margin-bottom: 6px;
  font-size: 14px;
}
.help-item span {
  color: #429dff;
  font-weight: bold;
}
.help-item p {
  margin: 0;
  color: #555;
}

.about-content {
  font-size: 13px;
  color: #666;
  line-height: 1.6;
}
.version {
  color: #999;
  margin-top: 4px;
}

.footer {
  text-align: center;
  font-size: 12px;
  color: #ccc;
  margin-top: 10px;
}
</style>