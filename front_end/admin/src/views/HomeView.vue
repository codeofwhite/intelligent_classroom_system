<template>
  <div class="teacher-home">
    <!-- 顶部欢迎 -->
    <div class="home-header">
      <h2>👋 您好，{{ teacherName }}！欢迎使用智慧课堂分析系统</h2>
      <p class="date">{{ today }}</p>
    </div>

    <!-- 数据统计卡片 -->
    <div class="data-grid">
      <div class="stat-card blue">
        <p class="label">今日课程</p>
        <h3>{{ todayCourse }}</h3>
        <span>节</span>
      </div>

      <div class="stat-card green">
        <p class="label">实时平均抬头率</p>
        <h3>{{ realTimeLookUp }}%</h3>
        <span>CV实时分析</span>
      </div>

      <div class="stat-card orange">
        <p class="label">课堂专注度</p>
        <h3>{{ focusRate }}%</h3>
        <span>综合评分</span>
      </div>

      <div class="stat-card red">
        <p class="label">异常行为学生</p>
        <h3>{{ abnormalStudents }}</h3>
        <span>走神/低头/讲话</span>
      </div>

      <div class="stat-card purple">
        <p class="label">班级总人数</p>
        <h3>{{ totalStudents }}</h3>
        <span>在线 {{ onlineStudents }} 人</span>
      </div>

      <div class="stat-card cyan">
        <p class="label">待推送报告</p>
        <h3>{{ waitPushReports }}</h3>
        <span>家长端可接收</span>
      </div>
    </div>

    <!-- 快捷功能入口 -->
    <div class="quick-bar">
      <h3>🚀 快捷操作</h3>
      <div class="quick-grid">
        <button class="quick-btn primary" @click="goToClass">
          <span>▶️</span>
          开始上课
        </button>
        <button class="quick-btn" @click="goToMember">
          <span>👥</span>
          会员管理
        </button>
        <button class="quick-btn" @click="goToClassGroup">
          <span>🏫</span>
          班级/小组
        </button>
        <button class="quick-btn" @click="goToReport">
          <span>📊</span>
          报告管理
        </button>
      </div>
    </div>

    <!-- 内容区域 -->
    <div class="action-sections">
      <!-- 实时课堂状态 -->
      <section class="action-block">
        <div class="block-header">
          <h3>📹 当前课堂实时状态</h3>
        </div>
        <div class="real-time-info">
          <div class="real-item">
            <label>当前班级</label>
            <span>{{ currentClass }}</span>
          </div>
          <div class="real-item">
            <label>上课时间</label>
            <span>{{ classTime }}</span>
          </div>
          <div class="real-item">
            <label>课程名称</label>
            <span>{{ className }}</span>
          </div>
        </div>
        <div class="tip">
          计算机视觉分析已启动，实时监测学生抬头率、专注度、行为状态
        </div>
      </section>

      <!-- 待处理报告 -->
      <section class="action-block">
        <div class="block-header">
          <h3>📝 待审阅行为报告</h3>
        </div>
        <table class="report-table">
          <thead>
            <tr>
              <th>学生</th>
              <th>班级</th>
              <th>分析类型</th>
              <th>状态</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="item in reportList" :key="item.id">
              <td>{{ item.name }}</td>
              <td>{{ item.class }}</td>
              <td>{{ item.type }}</td>
              <td>
                <span :class="['status', item.status === '待审阅' ? 'wait' : 'done']">
                  {{ item.status }}
                </span>
              </td>
              <td>
                <button class="action-btn" v-if="item.status === '待审阅'">审阅</button>
                <button class="action-btn success" v-else>已完成</button>
              </td>
            </tr>
          </tbody>
        </table>
      </section>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

// 老师信息
const teacherName = ref('王老师')
const today = ref('2025年12月19日 星期五')

// 数据统计
const todayCourse = ref(4)
const realTimeLookUp = ref(92)
const focusRate = ref(88)
const abnormalStudents = ref(3)
const totalStudents = ref(45)
const onlineStudents = ref(42)
const waitPushReports = ref(16)

// 当前课堂
const currentClass = ref('一年级1班')
const classTime = ref('09:00 ~ 09:40')
const className = ref('数学 · 分数乘除法')

// 报告列表
const reportList = ref([
  { id: 1, name: '张小明', class: '一年级1班', type: '课堂专注度', status: '待审阅' },
  { id: 2, name: '李华', class: '一年级1班', type: '抬头行为', status: '待审阅' },
  { id: 3, name: '王磊', class: '一年级1班', type: '课堂参与', status: '已审阅' },
])

// 跳转方法
const goToClass = () => alert('进入上课页面')
const goToMember = () => alert('跳转到会员管理')
const goToClassGroup = () => alert('跳转到班级小组管理')
const goToReport = () => alert('跳转到报告页面')
</script>

<style scoped>
.teacher-home {
  padding: 24px;
  background: #f5f7fa;
  min-height: 100vh;
}

.home-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.home-header h2 {
  margin: 0;
  font-size: 22px;
  color: #333;
}

.date {
  color: #666;
  font-size: 14px;
}

/* 数据卡片 */
.data-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
  margin-bottom: 24px;
}

.stat-card {
  background: #fff;
  padding: 20px;
  border-radius: 10px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
  position: relative;
  overflow: hidden;
}

.stat-card.blue { border-left: 4px solid #1890ff; }
.stat-card.green { border-left: 4px solid #52c41a; }
.stat-card.orange { border-left: 4px solid #faad14; }
.stat-card.red { border-left: 4px solid #f5222d; }
.stat-card.purple { border-left: 4px solid #722ed1; }
.stat-card.cyan { border-left: 4px solid #13c2c2; }

.stat-card .label {
  color: #666;
  margin: 0 0 8px 0;
  font-size: 14px;
}

.stat-card h3 {
  font-size: 30px;
  margin: 0 0 4px 0;
  color: #333;
}

.stat-card span {
  font-size: 12px;
  color: #999;
}

/* 快捷操作 */
.quick-bar {
  background: white;
  padding: 16px 20px;
  border-radius: 10px;
  margin-bottom: 24px;
}

.quick-bar h3 {
  margin: 0 0 16px 0;
  font-size: 16px;
}

.quick-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
}

.quick-btn {
  padding: 16px;
  border: none;
  border-radius: 8px;
  background: #f7f8fa;
  font-size: 14px;
  cursor: pointer;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
}

.quick-btn.primary {
  background: #1890ff;
  color: white;
}

.quick-btn span {
  font-size: 20px;
}

/* 内容区块 */
.action-sections {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

.action-block {
  background: white;
  padding: 20px;
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.block-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  padding-bottom: 10px;
  border-bottom: 1px solid #f0f0f0;
}

.block-header h3 {
  margin: 0;
  font-size: 16px;
}

/* 实时信息 */
.real-time-info {
  margin-bottom: 16px;
}

.real-item {
  display: flex;
  justify-content: space-between;
  padding: 10px 0;
  border-bottom: 1px dashed #eee;
}

.real-item label {
  color: #666;
  font-weight: 500;
}

.tip {
  background: #e6f7ff;
  color: #1890ff;
  padding: 10px;
  border-radius: 6px;
  font-size: 13px;
  margin-top: 10px;
}

/* 报告表格 */
.report-table {
  width: 100%;
  border-collapse: collapse;
}

.report-table th,
.report-table td {
  padding: 12px;
  text-align: left;
  border-bottom: 1px solid #f0f0f0;
}

.status {
  padding: 2px 8px;
  border-radius: 4px;
  font-size: 12px;
}

.status.wait {
  background: #fff7e6;
  color: #d48806;
}

.status.done {
  background: #f6ffed;
  color: #52c41a;
}

.action-btn {
  padding: 4px 12px;
  border: none;
  border-radius: 4px;
  background: #1890ff;
  color: white;
  cursor: pointer;
}

.action-btn.success {
  background: #52c41a;
}
</style>