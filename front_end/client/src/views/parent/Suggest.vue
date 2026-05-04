<template>
  <div class="advice-page">
    <div class="page-header">
      <h2>🤝 家校共育 · AI 成长分析</h2>
      <p>基于历史行为数据 · 智能生成个性化教育方案</p>
    </div>

    <!-- 选择孩子 -->
    <div class="select-box" v-if="studentList.length > 0">
      <label>选择孩子：</label>
      <select v-model="currentCode" @change="getAIAdvice">
        <option v-for="s in studentList" :key="s.student_code" :value="s.student_code">
          {{ s.student_name }}
        </option>
      </select>
    </div>

    <!-- 综合分析 -->
    <div class="card summary-card">
      <h3>📈 课堂行为综合分析</h3>
      <div class="text">{{ summary }}</div>
    </div>

    <!-- AI 共育建议 -->
    <div class="card ai-card">
      <h3>🤖 AI 个性化共育建议</h3>
      <div class="text">{{ advice }}</div>
    </div>

    <!-- 协同提示 -->
    <div class="card tip-card">
      <h3>📩 家校协同说明</h3>
      <div class="text">
        教师会定期同步课堂状态，家长可根据 AI 建议进行家庭引导，
        实现学校与家庭双向共育，帮助孩子持续提升课堂表现与学习习惯。
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

// 家长信息
let parentUser = null
try {
  parentUser = JSON.parse(localStorage.getItem('currentUser'))
} catch (e) {}

const studentList = ref([])
const currentCode = ref('')
const summary = ref('加载中...')
const advice = ref('加载中...')

// 获取孩子列表
onMounted(async () => {
  if (!parentUser || parentUser.role !== 'parent') return
  try {
    const res = await axios.post('http://localhost:5001/parent-children', {
      user_code: parentUser.user_code
    })
    studentList.value = res.data.children || []
    if (studentList.value.length > 0) {
      currentCode.value = studentList.value[0].student_code
      getAIAdvice()
    }
  } catch (err) {
    console.error(err)
  }
})

// 获取AI分析
async function getAIAdvice() {
  if (!currentCode.value) return
  try {
    const res = await axios.get('http://localhost:5002/api/ai/advice', {
      params: { student_code: currentCode.value }
    })
    summary.value = res.data.summary
    advice.value = res.data.advice
  } catch (err) {
    summary.value = '暂无数据'
    advice.value = '请稍后再试'
  }
}
</script>

<style scoped>
.advice-page {
  padding: 24px;
  background: linear-gradient(to bottom, #f9faff, #f1f5ff);
  min-height: 100vh;
  font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
}

.page-header {
  text-align: center;
  margin-bottom: 26px;
}
.page-header h2 {
  font-size: 24px;
  font-weight: 600;
  margin: 0 0 8px 0;
  color: #2c3e50;
}
.page-header p {
  font-size: 14px;
  color: #7f8c8d;
  margin: 0;
}

.select-box {
  text-align: center;
  margin-bottom: 24px;
}
.select-box label {
  margin-right: 8px;
  font-size: 15px;
  color: #555;
}
.select-box select {
  padding: 10px 14px;
  border-radius: 12px;
  border: 1px solid #e2e8ff;
  background: #fff;
  font-size: 14px;
}

.card {
  background: #fff;
  border-radius: 18px;
  padding: 22px;
  margin-bottom: 18px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}
.card h3 {
  margin: 0 0 14px 0;
  font-size: 17px;
  font-weight: 600;
  color: #2c3e50;
}
.text {
  line-height: 1.8;
  font-size: 15px;
  color: #444;
}

.summary-card {
  border-left: 5px solid #7c5fff;
}
.ai-card {
  border-left: 5px solid #50c878;
}
.tip-card {
  border-left: 5px solid #ffb946;
  background: #fffbf5;
}
</style>