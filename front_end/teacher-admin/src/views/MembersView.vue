<template>
  <div class="members-container">
    <h3>👨‍🎓 班级成员管理</h3>

    <div class="info-box">
      当前班级：<b>{{ className }}</b> | 学生总数：<b>{{ studentList.length }}</b>
    </div>

    <table class="members-table">
      <thead>
        <tr>
          <th>序号</th>
          <th>学生姓名</th>
          <th>性别</th>
          <th>班级</th>
          <th>绑定家长</th>
          <th>家长电话</th>
          <th>状态</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="(s, index) in studentList" :key="s.student_code">
          <td>{{ index + 1 }}</td>
          <td>{{ s.student_name }}</td>
          <td>{{ s.gender }}</td>
          <td>{{ className }}</td>
          <td>{{ s.parent_name || '未绑定' }}</td>
          <td>{{ s.parent_phone || '—' }}</td>
          <td><span class="status-normal">正常</span></td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const className = ref('')
const studentList = ref([])

const loadData = async () => {
  const userInfo = JSON.parse(localStorage.getItem('userInfo'))
  if (!userInfo) return

  try {
    const { data } = await axios.post('http://localhost:5001/teacher-students', {
      user_code: userInfo.user_code
    })

    className.value = data.class_name
    studentList.value = data.students
  } catch (err) {
    console.error(err)
  }
}

onMounted(() => loadData())
</script>

<style scoped>
.members-container {
  width: 95%;
  margin: 20px auto;
}

.info-box {
  background: #e6f7ff;
  padding: 12px 16px;
  border-radius: 8px;
  margin-bottom: 16px;
  font-size: 14px;
}

.members-table {
  width: 100%;
  border-collapse: collapse;
  background: #fff;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  overflow: hidden;
}

.members-table th {
  background: #f5f7fa;
  padding: 14px;
  text-align: left;
  font-weight: 600;
}

.members-table td {
  padding: 12px 14px;
  border-bottom: 1px solid #f0f0f0;
}

.status-normal {
  color: #52c41a;
  font-weight: bold;
}
</style>