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

    <!-- 今日课堂 -->
    <div class="today-card">
      <div class="card-header">
        <h3>📅 今日课堂</h3>
      </div>
      <div class="today-content">
        <div class="lesson-item">
          <div class="dot"></div>
          <div>
            <div class="lesson-name">数学 · 分数乘除法</div>
            <div class="lesson-time">第 3 节 | 09:50 ~ 10:35</div>
          </div>
        </div>
        <div class="tip">专注听讲，积极表现哦～</div>
      </div>
    </div>

    <!-- 快捷功能入口 -->
    <div class="grid-group">
      <div class="grid-item" @click="$router.push('/report')">
        <div class="icon">📊</div>
        <div class="title">行为报告</div>
        <div class="desc">查看课堂表现</div>
      </div>

      <div class="grid-item" @click="$router.push('/rank')">
        <div class="icon">🏆</div>
        <div class="title">专注排行</div>
        <div class="desc">班级荣誉榜</div>
      </div>

      <div class="grid-item" @click="$router.push('/medal')">
        <div class="icon">🎖️</div>
        <div class="title">我的勋章</div>
        <div class="desc">荣誉成长</div>
      </div>

      <div class="grid-item" @click="showNotice = true">
        <div class="icon">📢</div>
        <div class="title">系统公告</div>
        <div class="desc">最新通知</div>
      </div>
    </div>

    <!-- 使用帮助 -->
    <div class="section-card">
      <h3>📘 使用说明</h3>
      <div class="help-content">
        <div class="help-item">
          <span>1.</span>
          <p>每节课后自动生成课堂行为报告</p>
        </div>
        <div class="help-item">
          <span>2.</span>
          <p>专注度越高，可获得更多荣誉勋章</p>
        </div>
        <div class="help-item">
          <span>3.</span>
          <p>老师评语与AI建议会同步推送给你</p>
        </div>
      </div>
    </div>

    <!-- 关于系统 -->
    <div class="section-card dark">
      <h3>ℹ️ 关于系统</h3>
      <div class="about-content">
        <p>智慧课堂行为分析系统 · 学生端</p>
        <p>基于AI视觉分析 | 专注度监测 | 成长激励体系</p>
        <p class="version">v1.0.0</p>
      </div>
    </div>

    <!-- 底部 -->
    <div class="footer">
      智慧课堂 © 2025 毕业设计专用
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

const studentName = ref('张三')
const className = ref('一年级1班')
const dateText = ref('2026年04月20日 星期日')
const showNotice = ref(false)
</script>

<style scoped>
.student-home {
  padding: 20px;
  background: #f7f8fa;
  min-height: 100vh;
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
}

/* 问候卡片 */
.welcome-card {
  background: linear-gradient(90deg, #429dff, #57b9ff);
  color: white;
  border-radius: 16px;
  padding: 20px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.welcome-text h2 {
  margin: 0 0 4px 0;
  font-size: 20px;
}
.welcome-text p {
  margin: 0;
  font-size: 13px;
  opacity: 0.9;
}
.avatar {
  width: 50px;
  height: 50px;
  background: rgba(255, 255, 255, 0.2);
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 24px;
}

/* 今日课堂 */
.today-card {
  background: white;
  border-radius: 16px;
  padding: 18px;
  margin-bottom: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}
.card-header h3 {
  margin: 0 0 12px 0;
  font-size: 16px;
}
.lesson-item {
  display: flex;
  align-items: flex-start;
  gap: 10px;
  margin-bottom: 10px;
}
.dot {
  width: 8px;
  height: 8px;
  background: #429dff;
  border-radius: 50%;
  margin-top: 8px;
}
.lesson-name {
  font-weight: bold;
  font-size: 15px;
  margin-bottom: 4px;
}
.lesson-time {
  font-size: 12px;
  color: #999;
}
.tip {
  font-size: 12px;
  color: #888;
  margin-top: 8px;
}

/* 功能网格 */
.grid-group {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
  margin-bottom: 16px;
}
.grid-item {
  background: white;
  border-radius: 16px;
  padding: 18px 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
}
.grid-item .icon {
  font-size: 24px;
  margin-bottom: 8px;
}
.grid-item .title {
  font-weight: bold;
  font-size: 15px;
  margin-bottom: 4px;
}
.grid-item .desc {
  font-size: 12px;
  color: #999;
}

/* 区块卡片 */
.section-card {
  background: white;
  border-radius: 16px;
  padding: 18px;
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