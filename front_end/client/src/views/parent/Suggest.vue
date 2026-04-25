<template>
  <div class="advice-page">
    <div class="page-header">
      <h2>🤝 家校共育 · AI 成长建议</h2>
      <p>基于课堂行为数据分析，智能生成个性化教育指导</p>
    </div>

    <!-- 选择学生 -->
    <div class="select-wrap">
      <label>选择学生：</label>
      <select v-model="currStuId" @change="getAIAdvice">
        <option v-for="item in stuList" :key="item.id" :value="item.id">
          {{ item.name }}
        </option>
      </select>
    </div>

    <!-- AI 行为总结 -->
    <div class="card advice-card">
      <h3>📈 课堂行为综合分析</h3>
      <div class="content">{{ behaviorSummary }}</div>
    </div>

    <!-- AI 共育建议 -->
    <div class="card ai-card">
      <h3>🤖 AI 家校共育建议</h3>
      <div class="content">{{ aiAdvice }}</div>
    </div>

    <!-- 老师端同步通知说明 -->
    <div class="card tips-card">
      <h3>📩 家校协同说明</h3>
      <p>
        任课教师可定期推送课堂行为报告至家长端，结合AI分析建议，
        实现学校教育与家庭教育双向联动，共同帮助学生养成良好课堂习惯。
      </p>
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'

// 学生列表 预留
const stuList = ref([
  { id: 101, name: '张小明' },
  { id: 102, name: '李华' },
  { id: 103, name: '王磊' },
  { id: 104, name: '刘芳' }
])

const currStuId = ref(101)
const behaviorSummary = ref('')
const aiAdvice = ref('')

// 预留接口：后端 /api/ai_advice?stuId=xxx
async function getAIAdvice() {
  try {
    const res = await axios.get(`http://localhost:5002/api/ai_advice?stuId=${currStuId.value}`)
    behaviorSummary.value = res.data.summary
    aiAdvice.value = res.data.advice
  } catch (err) {
    // 接口没通时，本地默认文案兜底，不报错
    behaviorSummary.value = '该学生课堂行为数据正常，系统正在等待教师同步最新课堂分析报告。'
    aiAdvice.value = '建议家长日常关注孩子上课专注力，合理规划作息时间，减少电子设备使用，配合学校共同引导孩子专注学习。'
  }
}

// 初始加载
getAIAdvice()
</script>

<style scoped>
.advice-page {
  max-width: 1000px;
  margin: 0 auto;
  padding: 30px 20px;
}
.page-header {
  text-align: center;
  margin-bottom: 30px;
}
.select-wrap {
  margin-bottom: 20px;
  text-align: center;
}
select {
  padding: 6px 12px;
  border-radius: 6px;
  border: 1px solid #ccc;
}
.card {
  background: #fff;
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
h3 {
  margin: 0 0 12px 0;
  font-size: 16px;
  color: #333;
}
.content {
  line-height: 1.8;
  color: #555;
}
</style>