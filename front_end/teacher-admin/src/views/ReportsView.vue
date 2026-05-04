<template>
  <div class="report-page">
    <h2>📝 学生课堂行为报告</h2>

    <!-- 班级 + 时间 -->
    <div class="class-info">
      班级：
      <select v-model="classId" @change="loadStudents" style="padding: 6px 12px; border-radius: 6px;">
        <option value="">-- 选择班级 --</option>
        <option v-for="c in classList" :key="c.class_code" :value="c.class_code">
          {{ c.class_name }}
        </option>
      </select>

      &nbsp;&nbsp;
      课堂时间：
      <input v-model="lessonTime" type="datetime-local" style="padding: 6px 12px; border-radius: 6px;">
    </div>

    <!-- 学生列表 -->
    <div class="student-list" v-if="classId">
      <h4>选择学生</h4>
      <div class="grid">
        <button v-for="s in studentList" :key="s.student_code" class="student-btn"
          :class="{ active: selectedStudent?.student_code === s.student_code }" @click="selectStudent(s)">
          {{ s.name }}
        </button>
      </div>
    </div>

    <!-- 报告面板 -->
    <div class="report-card" v-if="selectedStudent && reportData">
      <h3>🎓 个人行为分析</h3>

      <div class="behavior">
        <div><label>正常坐姿：</label><span>{{ reportData.normal || 0 }}</span></div>
        <div><label>举手次数：</label><span>{{ reportData.raised_hand || 0 }}</span></div>
        <div><label>低头次数：</label><span>{{ reportData.looking_down || 0 }}</span></div>
        <div><label>专注度：</label><span>{{ focusRate }}%</span></div>
      </div>

      <div class="edit-section">
        <label>AI 自动评语</label>
        <textarea v-model="aiComment" placeholder="AI 分析结果"></textarea>

        <label>老师评分（1-100）</label>
        <input v-model="score" type="number" />

        <label>老师评语</label>
        <textarea v-model="teacherComment" placeholder="输入老师评语"></textarea>

        <div class="btns">
          <button class="btn ai" @click="runAI">AI 自动分析</button>
          <button class="btn save" @click="saveReport">保存报告</button>
        </div>
      </div>
    </div>
    <!-- 新增：该学生历史报告列表 -->
    <div class="history-card" v-if="selectedStudent && historyList.length > 0">
      <h3>📜 该学生历史课堂报告</h3>
      <div class="history-item" v-for="item in historyList" :key="item.id" @click="fillReport(item)">
        <div class="time">📅 {{ item.lesson_time }}</div>
        <div class="line">专注度：{{ item.focus_rate }}分｜老师评分：{{ item.teacher_score || '未评分' }}</div>
        <div class="comment">AI评语：{{ item.ai_comment ? item.ai_comment.slice(0, 30) + '...' : '无' }}</div>
      </div>
    </div>
    <div class="empty" v-else>
      请选择班级 → 选择学生
    </div>
  </div>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import axios from 'axios'

// 班级
const classList = ref([])
const classId = ref('')
const lessonTime = ref('')

// 学生
const studentList = ref([])
const selectedStudent = ref(null)

// 行为数据
const reportData = ref(null)
const aiComment = ref('')
const score = ref('')
const teacherComment = ref('')

const historyList = ref([])

// 专注度
const focusRate = computed(() => {
  const n = reportData.value?.normal || 0
  const r = reportData.value?.raised_hand || 0
  const d = reportData.value?.looking_down || 0
  const total = n + r + d
  if (total === 0) return 100
  return Math.round(((n + r) / total) * 100)
})

// 加载班级
const loadClasses = async () => {
  try {
    const res = await axios.get('http://localhost:5002/api/class/list')
    classList.value = res.data.list || []
  } catch (err) {
    console.error(err)
  }
}

// 加载学生
const loadStudents = async () => {
  if (!classId.value) return
  try {
    const res = await axios.get('http://localhost:5002/api/class/students', {
      params: { class_code: classId.value }
    })
    studentList.value = res.data.students || []
  } catch (err) {
    console.error(err)
  }
}

// 选择学生 → 加载行为 + 加载历史报告
const selectStudent = async (stu) => {
  selectedStudent.value = stu
  aiComment.value = ''
  score.value = ''
  teacherComment.value = ''
  historyList.value = []  // 清空历史

  try {
    const faceRes = await axios.get('http://localhost:5002/api/face/by_student', {
      params: { student_code: stu.student_code }
    })
    const faceId = faceRes.data.face_id

    if (!faceId) {
      alert('未绑定人脸')
      reportData.value = null
      return
    }

    const reportRes = await axios.get('http://localhost:5002/api/student/behavior', {
      params: { class_code: classId.value, face_id: faceId }
    })

    const b = reportRes.data.behaviors || {}
    reportData.value = {
      normal: b['正常坐姿'] || 0,
      raised_hand: b['举手'] || 0,
      looking_down: b['低头'] || 0
    }

    // 🔥 加载该学生历史报告
    await loadHistoryReport(stu.student_code)

  } catch (err) {
    console.error(err)
  }
}

// 加载历史报告
const loadHistoryReport = async (studentCode) => {
  try {
    const res = await axios.get('http://localhost:5002/api/report/history', {
      params: {
        student_code: studentCode,
        class_code: classId.value
      }
    })
    historyList.value = res.data.list || []
  } catch (err) {
    console.error(err)
  }
}

// 点击历史记录，回填到编辑框
const fillReport = (item) => {
  lessonTime.value = item.lesson_time.slice(0, 16)
  aiComment.value = item.ai_comment || ''
  score.value = item.teacher_score || ''
  teacherComment.value = item.teacher_comment || ''
}

// ======================
// AI 分析（后端调用）
// ======================
const runAI = async () => {
  if (!reportData.value) return
  try {
    const res = await axios.post('http://localhost:5002/api/ai/analyze', {
      student_code: selectedStudent.value.student_code,
      normal_posture: reportData.value.normal,
      raised_hand: reportData.value.raised_hand,
      looking_down: reportData.value.looking_down,
      focus_rate: focusRate.value
    })
    aiComment.value = res.data.comment
  } catch (err) {
    alert('AI 分析失败')
  }
}

// ======================
// 保存报告（MySQL）
// ======================
const saveReport = async () => {
  if (!lessonTime.value) {
    alert('请选择课堂时间！')
    return
  }

  try {
    await axios.post('http://localhost:5002/api/report/save', {
      student_code: selectedStudent.value.student_code, // ✅ 正确
      class_code: classId.value,
      lesson_time: lessonTime.value,
      normal_posture: reportData.value.normal,
      raised_hand: reportData.value.raised_hand,
      looking_down: reportData.value.looking_down,
      focus_rate: focusRate.value,
      ai_comment: aiComment.value,
      teacher_score: score.value,
      teacher_comment: teacherComment.value
    })
    alert('✅ 报告保存成功！')
  } catch (err) {
    console.error(err)
    alert('保存失败')
  }
}

onMounted(() => {
  loadClasses()
})
</script>

<style scoped>
.report-page {
  padding: 24px;
}

.class-info {
  background: #e6f7ff;
  padding: 12px 16px;
  border-radius: 8px;
  margin-bottom: 20px;
}

.student-list h4 {
  margin-bottom: 12px;
}

.grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 10px;
  margin-bottom: 24px;
}

.student-btn {
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 6px;
  background: #fff;
  cursor: pointer;
}

.student-btn.active {
  background: #1890ff;
  color: #fff;
  border-color: #1890ff;
}

.report-card {
  background: #fff;
  padding: 24px;
  border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.behavior {
  background: #f5f7fa;
  padding: 16px;
  border-radius: 8px;
  margin-bottom: 20px;
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}

.behavior div {
  display: flex;
  justify-content: space-between;
}

.edit-section {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.edit-section input,
.edit-section textarea {
  padding: 10px;
  border: 1px solid #ddd;
  border-radius: 6px;
}

.edit-section textarea {
  height: 100px;
  resize: none;
}

.btns {
  display: flex;
  gap: 12px;
  margin-top: 10px;
}

.btn {
  padding: 10px 16px;
  border: none;
  border-radius: 6px;
  cursor: pointer;
}

.btn.ai {
  background: #722ed1;
  color: #fff;
}

.btn.save {
  background: #1890ff;
  color: #fff;
}

.history-card {
  background: #fff;
  padding: 20px;
  border-radius: 12px;
  margin-top: 20px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
}

.history-item {
  border: 1px solid #eee;
  border-radius: 8px;
  padding: 12px;
  margin-bottom: 10px;
  cursor: pointer;
}

.history-item:hover {
  background: #f8f9fa;
}

.time {
  color: #666;
  font-size: 14px;
  margin-bottom: 4px;
}

.line {
  margin: 4px 0;
}

.comment {
  color: #888;
  font-size: 14px;
}

.empty {
  padding: 40px;
  text-align: center;
  color: #999;
}
</style>