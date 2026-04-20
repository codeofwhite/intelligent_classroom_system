<template>
  <div class="student-rank-page">
    <!-- 顶部标题 -->
    <div class="header">
      <h2>🏆 课堂专注成长档案</h2>
      <p class="subtitle">看得见的进步，看得见的努力</p>
    </div>

    <!-- 激励卡片 -->
    <div class="encourage-card">
      <div class="icon">💡</div>
      <p class="text">{{ encourageText }}</p>
    </div>

    <!-- 我的总体数据 -->
    <div class="my-overview">
      <div class="my-info">
        <div class="avatar">👦</div>
        <div>
          <h4>{{ studentName }}</h4>
          <p>{{ className }}</p>
        </div>
      </div>
      <div class="total-score">
        <span class="num">{{ totalFocusScore }}</span>
        <p>平均专注度</p>
      </div>
    </div>

    <!-- 时间维度切换 -->
    <div class="tab-bar">
      <div
        v-for="tab in tabs"
        :key="tab.key"
        class="tab"
        :class="{ active: currentTab === tab.key }"
        @click="currentTab = tab.key"
      >
        {{ tab.label }}
      </div>
    </div>

    <!-- 今日 -->
    <div v-if="currentTab === 'day'" class="rank-section">
      <h3>📅 今日课堂表现</h3>
      <div class="card">
        <div class="item">
          <span>专注度</span>
          <span class="val">{{ dayData.focus }}%</span>
        </div>
        <div class="item">
          <span>抬头次数</span>
          <span class="val">{{ dayData.lookUp }}</span>
        </div>
        <div class="item">
          <span>走神次数</span>
          <span class="val">{{ dayData.disturb }}</span>
        </div>
      </div>
    </div>

    <!-- 本周 -->
    <div v-if="currentTab === 'week'" class="rank-section">
      <h3>📆 本周专注趋势</h3>
      <div class="card">
        <div class="item">
          <span>本周平均</span>
          <span class="val high">{{ weekData.avg }}%</span>
        </div>
        <div class="item">
          <span>较上周变化</span>
          <span class="val up">↑ {{ weekData.up }}%</span>
        </div>
        <div class="item">
          <span>最佳日</span>
          <span class="val">{{ weekData.bestDay }}</span>
        </div>
      </div>
    </div>

    <!-- 本月 -->
    <div v-if="currentTab === 'month'" class="rank-section">
      <h3>🗓️ 月度学习报告</h3>
      <div class="card">
        <div class="item">
          <span>月度平均专注</span>
          <span class="val high">{{ monthData.avg }}%</span>
        </div>
        <div class="item">
          <span>进步幅度</span>
          <span class="val up">↑ {{ monthData.progress }}%</span>
        </div>
        <div class="item">
          <span>出勤课堂</span>
          <span class="val">{{ monthData.classCount }} 节</span>
        </div>
      </div>
    </div>

    <!-- 本学期 -->
    <div v-if="currentTab === 'semester'" class="rank-section">
      <h3>🎖️ 本学期荣誉勋章</h3>
      <div class="medal-wrap">
        <div class="medal" v-for="m in semesterMedals" :key="m">
          <div class="icon">{{ m.icon }}</div>
          <p>{{ m.label }}</p>
        </div>
      </div>
      <div class="card" style="margin-top:12px">
        <div class="item">
          <span>学期总平均</span>
          <span class="val high">{{ semesterData.avg }}%</span>
        </div>
        <div class="item">
          <span>综合评级</span>
          <span class="val level">{{ semesterData.level }}</span>
        </div>
      </div>
    </div>

    <!-- 班级排行（固定展示） -->
    <div class="rank-section">
      <h3>📊 班级专注度排行</h3>
      <div class="rank-list">
        <div
          v-for="(item, idx) in rankList"
          :key="item.id"
          class="rank-item"
          :class="{ me: item.isMe }"
        >
          <div class="num">
            <span v-if="idx===0" class="g">1</span>
            <span v-else-if="idx===1" class="s">2</span>
            <span v-else-if="idx===2" class="b">3</span>
            <span v-else>{{ idx+1 }}</span>
          </div>
          <div class="name">{{ item.name }}</div>
          <div class="score">{{ item.score }}分</div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'

// 学生信息
const studentName = ref('张三')
const className = ref('一年级1班')
const totalFocusScore = ref(91)

// 激励语
const encourageText = ref('每一次专注，都在靠近更好的自己！')

// tab 切换
const currentTab = ref('day')
const tabs = ref([
  { key: 'day', label: '今日' },
  { key: 'week', label: '本周' },
  { key: 'month', label: '本月' },
  { key: 'semester', label: '本学期' }
])

// 模拟数据
const dayData = ref({ focus: 94, lookUp: 16, disturb: 2 })
const weekData = ref({ avg: 92, up: 5, bestDay: '周四' })
const monthData = ref({ avg: 90, progress: 7, classCount: 22 })
const semesterData = ref({ avg: 89, level: 'A · 优秀' })
const semesterMedals = ref([
  { icon: '🏅', label: '全勤标兵' },
  { icon: '🎯', label: '专注之星' },
  { icon: '📈', label: '进步先锋' },
  { icon: '🌟', label: '课堂模范' }
])

// 班级排行
const rankList = ref([
  { id: 1, name: '李华', score: 98, isMe: false },
  { id: 2, name: '张三', score: 93, isMe: true },
  { id: 3, name: '王磊', score: 90, isMe: false },
  { id: 4, name: '刘芳', score: 87, isMe: false },
  { id: 5, name: '陈明', score: 85, isMe: false },
])
</script>

<style scoped>
.student-rank-page {
  padding: 20px;
  background: #f7f8fa;
  min-height: 100vh;
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
}

.header {
  text-align: center;
  margin-bottom: 14px;
}
.header h2 {
  font-size: 22px;
  margin: 0 0 4px 0;
  color: #222;
}
.subtitle {
  font-size: 14px;
  color: #888;
  margin: 0;
}

/* 激励卡片 */
.encourage-card {
  background: linear-gradient(90deg, #429dff, #57b9ff);
  color: #fff;
  padding: 14px 16px;
  border-radius: 14px;
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}
.encourage-card .icon {
  font-size: 24px;
}
.encourage-card .text {
  margin: 0;
  font-size: 15px;
  line-height: 1.4;
}

/* 我的信息 */
.my-overview {
  background: #fff;
  border-radius: 14px;
  padding: 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.04);
}
.my-info {
  display: flex;
  align-items: center;
  gap: 12px;
}
.avatar {
  width: 42px;
  height: 42px;
  background: #e6f0ff;
  border-radius: 50%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 20px;
}
.my-info h4 {
  margin: 0 0 2px 0;
  font-size: 16px;
  color: #222;
}
.my-info p {
  margin: 0;
  font-size: 13px;
  color: #999;
}
.total-score {
  text-align: center;
}
.total-score .num {
  font-size: 26px;
  font-weight: bold;
  color: #429dff;
}
.total-score p {
  margin: 2px 0 0 0;
  font-size: 13px;
  color: #999;
}

/* tab 切换 */
.tab-bar {
  display: flex;
  background: #fff;
  border-radius: 10px;
  padding: 4px;
  margin-bottom: 16px;
  gap: 4px;
}
.tab {
  flex: 1;
  text-align: center;
  padding: 8px 0;
  border-radius: 8px;
  font-size: 14px;
  color: #888;
  cursor: pointer;
}
.tab.active {
  background: #429dff;
  color: #fff;
}

/* 内容区块 */
.rank-section h3 {
  font-size: 16px;
  margin: 0 0 8px 4px;
  color: #333;
}
.card {
  background: #fff;
  border-radius: 12px;
  padding: 14px 16px;
}
.card .item {
  display: flex;
  justify-content: space-between;
  padding: 10px 0;
  border-bottom: 1px solid #f2f2f2;
}
.card .item:last-child {
  border-bottom: none;
}
.item .val {
  font-weight: bold;
  color: #555;
}
.val.high { color: #429dff; }
.val.up { color: #20c997; }
.val.level { color: #fa8c16; }

/* 勋章 */
.medal-wrap {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
  margin-bottom: 10px;
}
.medal {
  background: #fff;
  border-radius: 10px;
  padding: 12px 6px;
  text-align: center;
}
.medal .icon {
  font-size: 22px;
  margin-bottom: 4px;
}
.medal p {
  margin: 0;
  font-size: 12px;
  color: #666;
}

/* 班级排行 */
.rank-list {
  background: #fff;
  border-radius: 12px;
  overflow: hidden;
}
.rank-item {
  display: flex;
  padding: 12px 16px;
  align-items: center;
  border-bottom: 1px solid #f5f5f5;
}
.rank-item:last-child { border: none; }
.rank-item.me { background: #f0f7ff; }
.rank-item .num {
  width: 26px;
  text-align: center;
  font-weight: bold;
  font-size: 15px;
}
.num .g { color: #ffb400; }
.num .s { color: #c0c0c0; }
.num .b { color: #e57c2a; }
.rank-item .name {
  flex: 1;
  margin-left: 8px;
  font-size: 15px;
}
.rank-item .score {
  font-weight: bold;
  color: #429dff;
  font-size: 15px;
}
</style>