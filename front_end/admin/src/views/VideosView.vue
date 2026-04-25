<template>
  <div class="video-analysis-page">
    <div class="header">
      <h2>🎓 智能课堂行为分析系统</h2>
      <p>按班级、课程、时间管理课堂分析记录</p>
    </div>

    <!-- 上传分析 -->
    <div class="card">
      <h3>📂 上传课堂视频进行分析</h3>
      <div class="upload-form">

        <!-- 🔥 新增：班级、日期、节次 -->
        <select v-model="classId" class="select-input">
          <option value="1">高一(1)班</option>
          <option value="2">高一(2)班</option>
          <option value="3">高一(3)班</option>
        </select>

        <input v-model="lessonSection" class="select-input" placeholder="节次：如 第1节" />
        <input v-model="recordDate" type="date" class="select-input" />

        <input type="file" ref="videoInput" @change="onFileChange" accept="video/*" />

        <select v-model="selectedModel" class="select-input" @change="switchModel" :disabled="switching">
          <option v-for="model in modelOptions" :key="model" :value="model">
            {{ model }}
          </option>
        </select>

        <button @click="startAnalysis" :disabled="loading || switching" class="btn-primary">
          {{ loading ? "AI 分析中..." : "开始上传并分析" }}
        </button>
      </div>

      <!-- 结果 -->
      <div v-if="resultUrl || statistics" class="result-section">
        <h4>✅ 分析完成</h4>
        <video v-if="resultUrl" :src="resultUrl" controls class="result-video"></video>

        <div v-if="statistics" class="stats-section">
          <h4>📊 课堂行为统计</h4>
          <table class="behavior-table">
            <tr>
              <th>行为</th>
              <th>次数</th>
            </tr>
            <tr v-for="(count, name) in statistics.behavior_counts" :key="name">
              <td>{{ name }}</td>
              <td>{{ count }}</td>
            </tr>
          </table>
        </div>
      </div>
    </div>

    <!-- 🔥 历史记录 → 班级+日期 卡片式折叠面板 -->
    <div class="card">
      <h3>📜 课堂分析历史记录</h3>
      <button @click="fetchTeacherReports" class="btn-sm" style="margin-left:10px;">
        🔄 刷新历史记录
      </button>
      <div v-for="group in dateClassGroups" :key="group.key" class="collapse-card">
        <div class="collapse-header" @click="toggleGroup(group.key)">
          <span>📅 {{ group.date }} | {{ group.className }}</span>
          <span>{{ group.list.length }} 条记录</span>
        </div>

        <div v-show="group.open" class="collapse-body">
          <div v-for="item in group.list" :key="item.id" class="lesson-item">
            <div>
              <div class="label">课程：{{ item.lesson_section }}</div>
              <div class="time">{{ item.created_at }}</div>
            </div>
            <button class="btn-sm" @click="goToAnalysis(item)">查看分析报告</button>
          </div>
        </div>
      </div>

      <div v-if="Object.keys(dateClassGroups).length === 0" class="empty">
        暂无分析记录
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import axios from 'axios'
import { useRouter } from 'vue-router'

const router = useRouter()
const videoInput = ref(null)
const videoFile = ref(null)
const loading = ref(false)
const switching = ref(false)
const resultUrl = ref('')
const statistics = ref(null)

const modelOptions = ref([])
const selectedModel = ref('')

// 🔥 新增：班级、日期、节次、教师
const classId = ref('1')
const lessonSection = ref('第1节')
const recordDate = ref('')
const teacherCode = ref('T2025001') // 登录后从用户信息取

const reportList = ref([])
const openGroups = ref({})

// 🔥 按【日期 + 班级】分组（卡片折叠面板）
const dateClassGroups = computed(() => {
  const groups = {}
  reportList.value.forEach(r => {
    const date = r.created_at.split(' ')[0]
    const key = `${date}_${r.class_id}`
    if (!groups[key]) {
      groups[key] = {
        key,
        date,
        classId: r.class_id,
        className: r.class_name,
        list: [],
        open: openGroups.value[key] || false
      }
    }
    groups[key].list.push(r)
  })
  return groups
})

function toggleGroup(key) {
  openGroups.value[key] = !openGroups.value[key]
}

// 读取老师的所有课程报告（修改版）
async function fetchTeacherReports() {
  try {
    const res = await axios.get('http://localhost:5002/api/teacher/reports', {
      params: {
        teacher_code: teacherCode.value,
        _t: Date.now() // 加时间戳，彻底绕过浏览器缓存
      },
      headers: {
        'Cache-Control': 'no-cache, no-store, must-revalidate', // 禁用所有缓存
        'Pragma': 'no-cache',
        'Expires': '0'
      }
    })
    reportList.value = res.data
    console.log("✅ 加载最新报告（无缓存）：", reportList.value)
  } catch (err) {
    console.error("加载报告失败", err)
    reportList.value = [] // 失败直接清空，不显示旧数据
  }
}

// 上传
function onFileChange(e) {
  videoFile.value = e.target.files[0]
}

async function startAnalysis() {
  if (!videoFile.value) return alert('请选择视频')
  loading.value = true
  const fd = new FormData()
  fd.append('video', videoFile.value)
  fd.append('teacher_code', teacherCode.value)
  fd.append('class_id', classId.value)
  fd.append('lesson_section', lessonSection.value)

  try {
    const res = await axios.post('http://localhost:5002/upload_video', fd)
    statistics.value = res.data.statistics
    alert('上传分析完成')
    fetchTeacherReports()
  } finally {
    loading.value = false
  }
}

// 进入深度分析页面
function goToAnalysis(item) {
  router.push({
    path: '/analysis-detail',
    query: { reportId: item.id }
  })
}

// 模型...
async function fetchModels() {
  const r = await axios.get('http://localhost:5002/get_models')
  modelOptions.value = r.data.models
  selectedModel.value = r.data.current
}

async function switchModel() {
  switching.value = true
  await axios.post('http://localhost:5002/switch_model', {
    model_name: selectedModel.value
  })
  switching.value = false
}

onMounted(() => {
  fetchModels()
  fetchTeacherReports()
})
</script>

<style scoped>
.video-analysis-page {
  max-width: 1200px;
  margin: 0 auto;
  padding: 30px 20px;
  background: #f7f9fc;
  min-height: 100vh;
}

.header {
  text-align: center;
  margin-bottom: 30px;
}

.card {
  background: white;
  border-radius: 16px;
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.upload-form {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  align-items: center;
}

.select-input {
  padding: 8px 12px;
  border: 1px solid #ddd;
  border-radius: 8px;
}

.btn-primary {
  background: #42b983;
  color: white;
  border: none;
  padding: 10px 18px;
  border-radius: 8px;
  cursor: pointer;
}

.result-video {
  width: 100%;
  border-radius: 10px;
  margin-top: 10px;
}

.behavior-table {
  width: 100%;
  border-collapse: collapse;
  margin-top: 10px;
}

.behavior-table th,
td {
  border: 1px solid #eee;
  padding: 10px;
  text-align: center;
}

/* 折叠卡片 */
.collapse-card {
  margin-bottom: 12px;
  border: 1px solid #eee;
  border-radius: 12px;
  overflow: hidden;
}

.collapse-header {
  padding: 14px 18px;
  background: #f9fafb;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  font-weight: 500;
}

.collapse-body {
  padding: 14px 18px;
}

.lesson-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 10px 0;
  border-bottom: 1px solid #f0f0f0;
}

.btn-sm {
  padding: 6px 12px;
  background: #e6f7ff;
  border: 1px solid #91d5ff;
  border-radius: 6px;
  cursor: pointer;
}

.time {
  font-size: 12px;
  color: #999;
}

.empty {
  text-align: center;
  padding: 30px;
  color: #999;
}
</style>