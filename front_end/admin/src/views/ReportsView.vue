<template>
  <div class="report-page">
    <h2>📝 学生行为报告管理</h2>

    <!-- 班级信息 -->
    <div class="class-info">
      班级：<b>{{ className }}</b>
      学生总数：<b>{{ studentList.length }}</b>
    </div>

    <!-- 学生列表（选择学生查看报告） -->
    <div class="student-list">
      <h4>选择学生</h4>
      <div class="grid">
        <button v-for="s in studentList" :key="s.student_id" class="student-btn"
          :class="{ active: selectedStudent?.student_id === s.student_id }" @click="selectStudent(s)">
          {{ s.student_name }}（{{ s.gender }}）
        </button>
      </div>
    </div>

    <!-- 报告区域（选中学生才显示） -->
    <div class="report-card" v-if="selectedStudent">
      <h3>🎓 行为分析报告</h3>

      <div class="info">
        <div>学生：{{ selectedStudent.student_name }}</div>
        <div>班级：{{ className }}</div>
        <div>日期：{{ today }}</div>
      </div>

      <div class="behavior">
        <div><label>抬头率：</label><span>—</span></div>
        <div><label>专注度：</label><span>—</span></div>
        <div><label>举手次数：</label><span>—</span></div>
        <div><label>异常行为：</label><span>—</span></div>
      </div>

      <div class="edit-section">
        <label>老师评分</label>
        <input v-model="score" type="number" placeholder="1-100" />

        <label>老师评语</label>
        <textarea v-model="comment" placeholder="输入课堂表现评语"></textarea>

        <div class="btns">
          <button class="btn save">保存报告</button>
          <button class="btn push">推送给家长</button>
          <button class="btn ai">AI 自动分析</button>
        </div>
      </div>
    </div>

    <div class="empty" v-else>
      请选择左侧学生查看/编写行为报告
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const className = ref('')
const studentList = ref([])
const selectedStudent = ref(null)
const score = ref('')
const comment = ref('')
const today = ref('')

// 日期
const getDate = () => {
  const d = new Date()
  return `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`
}

// 加载学生
const loadStudents = async () => {
  const user = JSON.parse(localStorage.getItem('userInfo'))
  if (!user) return

  try {
    const { data } = await axios.post('http://localhost:5001/teacher-students', {
      user_id: user.id
    })
    className.value = data.class_name
    studentList.value = data.students
  } catch (err) {
    console.error(err)
  }
}

// 选择学生
const selectStudent = (s) => {
  selectedStudent.value = s
  score.value = ''
  comment.value = ''
}

onMounted(() => {
  today.value = getDate()
  loadStudents()
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

.info {
  display: flex;
  gap: 16px;
  margin-bottom: 20px;
  color: #666;
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

.btn.save {
  background: #1890ff;
  color: #fff;
}

.btn.push {
  background: #52c41a;
  color: #fff;
}

.btn.ai {
  background: #722ed1;
  color: #fff;
}

.empty {
  padding: 40px;
  text-align: center;
  color: #999;
}
</style>