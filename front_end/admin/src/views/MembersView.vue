<template>
  <div class="member-wrapper">
    <div class="member-card">
      <div class="card-header">
        <h3>👨‍🎓 班级成员管理</h3>
        <div class="class-info">
          当前班级：<span>{{ className }}</span> · 学生总数：<span>{{ studentList.length }}</span>
        </div>
      </div>

      <div class="table-box">
        <table class="member-table">
          <thead>
            <tr>
              <th>序号</th>
              <th>学生姓名</th>
              <th>学号</th>
              <th>性别</th>
              <th>人脸照片</th>
              <th>绑定家长</th>
              <th>家长电话</th>
              <th>操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="(s, index) in studentList" :key="s.student_code">
              <td>{{ index + 1 }}</td>
              <td class="student-name">{{ s.student_name }}</td>
              <td>{{ s.student_code }}</td>
              <td>{{ s.gender }}</td>
              <td>
                <span v-if="faceCountMap[s.student_code] > 0" class="face-badge has-face">
                  📷 {{ faceCountMap[s.student_code] }}张
                </span>
                <span v-else class="face-badge no-face">未上传</span>
              </td>
              <td>{{ s.parent_name || '未绑定' }}</td>
              <td>{{ s.parent_phone || '—' }}</td>
              <td>
                <button class="btn-face" @click="openFaceDialog(s)">📷 管理人脸</button>
              </td>
            </tr>
          </tbody>
        </table>

        <div class="empty" v-if="studentList.length === 0">
          🧑‍🎓 暂无学生数据
        </div>
      </div>
    </div>

    <!-- 人脸管理对话框 -->
    <div class="face-dialog-mask" v-if="faceDialogVisible" @click.self="faceDialogVisible = false">
      <div class="face-dialog">
        <div class="face-dialog-header">
          <h3>📷 人脸照片管理 - {{ faceTarget.student_name }}</h3>
          <button class="close-btn" @click="faceDialogVisible = false">✕</button>
        </div>

        <div class="face-dialog-body">
          <!-- 上传区域 -->
          <div class="upload-section">
            <p>上传人脸照片（建议正面清晰照片，支持多张）</p>
            <div class="upload-actions">
              <input type="file" ref="photoInput" @change="onPhotoChange" accept="image/*" multiple style="display:none" />
              <button class="btn-upload" @click="$refs.photoInput.click()">📁 选择照片</button>
              <button class="btn-upload-confirm" @click="uploadPhoto" :disabled="!photoFile || uploading">
                {{ uploading ? '上传中...' : '⬆️ 上传' }}
              </button>
            </div>
            <p class="upload-tip" v-if="photoFile">已选择：{{ photoFile.name }}</p>
          </div>

          <!-- 已有照片列表 -->
          <div class="photo-list">
            <h4>已上传的照片（{{ faceImages.length }}张）</h4>
            <div v-if="faceImages.length === 0" class="no-photo">暂无照片</div>
            <div class="photo-grid">
              <div v-for="img in faceImages" :key="img.id" class="photo-item">
                <img :src="img.url" class="photo-img" />
                <button class="photo-delete" @click="deleteImage(img)">🗑️</button>
              </div>
            </div>
          </div>

          <!-- 刷新缓存 -->
          <div class="refresh-section">
            <button class="btn-refresh" @click="refreshCache" :disabled="refreshing">
              {{ refreshing ? '刷新中...' : '🔄 同步到识别服务' }}
            </button>
            <p class="refresh-tip">上传/删除照片后，点击此按钮将变更同步到人脸识别和行为分析服务</p>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'

const MODEL_SERVICE = 'http://localhost:5002'
const USER_CENTER = 'http://localhost:5001'

const className = ref('')
const classCode = ref('')
const studentList = ref([])
const faceCountMap = ref({})

// 人脸管理对话框
const faceDialogVisible = ref(false)
const faceTarget = ref({})
const faceImages = ref([])
const photoFile = ref(null)
const uploading = ref(false)
const refreshing = ref(false)

const loadData = async () => {
  const userInfo = JSON.parse(localStorage.getItem('userInfo'))
  if (!userInfo) return

  try {
    const { data } = await axios.post(`${USER_CENTER}/teacher-students`, {
      user_code: userInfo.user_code
    })
    className.value = data.class_name
    classCode.value = data.class_code || ''
    studentList.value = data.students || []

    console.log('加载班级数据:', classCode.value, className.value)

    // 批量获取人脸照片数量
    await loadFaceCounts()
  } catch (err) {
    console.error(err)
  }
}

const loadFaceCounts = async () => {
  const counts = {}
  for (const s of studentList.value) {
    try {
      const { data } = await axios.get(`${MODEL_SERVICE}/api/face/images`, {
        params: { student_code: s.student_code }
      })
      counts[s.student_code] = (data.images || []).length
    } catch {
      counts[s.student_code] = 0
    }
  }
  faceCountMap.value = counts
}

const openFaceDialog = async (student) => {
  faceTarget.value = student
  faceDialogVisible.value = true
  photoFile.value = null
  await loadFaceImages(student.student_code)
}

const loadFaceImages = async (studentCode) => {
  try {
    const { data } = await axios.get(`${MODEL_SERVICE}/api/face/images`, {
      params: { student_code: studentCode }
    })
    faceImages.value = data.images || []
  } catch {
    faceImages.value = []
  }
}

const onPhotoChange = (e) => {
  photoFile.value = e.target.files[0]
}

const uploadPhoto = async () => {
  if (!photoFile.value) return
  if (!faceTarget.value.student_code) {
    alert('学生学号缺失，请刷新页面重试')
    return
  }
  if (!classCode.value) {
    alert('班级编号未加载，请刷新页面重试')
    return
  }

  uploading.value = true

  try {
    const fd = new FormData()
    fd.append('photo', photoFile.value)
    fd.append('student_code', String(faceTarget.value.student_code))
    fd.append('class_code', String(classCode.value))

    console.log('上传参数:', faceTarget.value.student_code, classCode.value)

    await axios.post(`${MODEL_SERVICE}/api/face/upload_image`, fd)
    photoFile.value = null
    await loadFaceImages(faceTarget.value.student_code)
    faceCountMap.value[faceTarget.value.student_code] = faceImages.value.length
    alert('上传成功！')
  } catch (err) {
    alert('上传失败：' + (err.response?.data?.msg || err.message))
  } finally {
    uploading.value = false
  }
}

const deleteImage = async (img) => {
  if (!confirm('确定删除这张照片？')) return
  try {
    await axios.post(`${MODEL_SERVICE}/api/face/delete_image`, { id: img.id })
    await loadFaceImages(faceTarget.value.student_code)
    faceCountMap.value[faceTarget.value.student_code] = faceImages.value.length
  } catch {
    alert('删除失败')
  }
}

const refreshCache = async () => {
  refreshing.value = true
  try {
    const { data } = await axios.post(`${MODEL_SERVICE}/api/face/refresh_cache`)
    alert('缓存刷新完成！\n' + JSON.stringify(data.results, null, 2))
  } catch (err) {
    alert('刷新失败：' + (err.response?.data?.msg || err.message))
  } finally {
    refreshing.value = false
  }
}

onMounted(() => loadData())
</script>

<style scoped>
.member-wrapper {
  width: 100%;
  padding: 24px;
  box-sizing: border-box;
  background: #f5f7fa;
  min-height: 100vh;
}

.member-card {
  background: #fff;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
  overflow: hidden;
}

.card-header {
  padding: 20px 24px;
  border-bottom: 1px solid #f0f2f5;
}

.card-header h3 {
  margin: 0 0 10px 0;
  font-size: 18px;
  font-weight: 600;
  color: #1d2129;
}

.class-info {
  font-size: 14px;
  color: #4e5969;
}

.class-info span {
  color: #1677ff;
  font-weight: 500;
}

.table-box {
  padding: 16px 24px 24px;
}

.member-table {
  width: 100%;
  border-collapse: collapse;
}

.member-table thead th {
  text-align: left;
  padding: 14px 12px;
  font-size: 14px;
  font-weight: 600;
  color: #4e5969;
  background: #fafbfc;
  border-bottom: 1px solid #e5e6eb;
}

.member-table tbody td {
  padding: 14px 12px;
  font-size: 14px;
  color: #1d2129;
  border-bottom: 1px solid #f0f2f5;
}

.member-table tbody tr:hover {
  background: #fafbfc;
  transition: background 0.2s;
}

.student-name {
  font-weight: 500;
  color: #1677ff;
}

.face-badge {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 12px;
}

.face-badge.has-face {
  background: #e6ffed;
  color: #00b42a;
}

.face-badge.no-face {
  background: #fff7e6;
  color: #fa8c16;
}

.btn-face {
  background: #e6f7ff;
  border: 1px solid #91d5ff;
  color: #1677ff;
  padding: 4px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 12px;
}

.btn-face:hover {
  background: #bae7ff;
}

.empty {
  text-align: center;
  padding: 30px;
  color: #999;
}

/* 人脸管理对话框 */
.face-dialog-mask {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.face-dialog {
  background: #fff;
  border-radius: 12px;
  width: 600px;
  max-height: 80vh;
  overflow-y: auto;
}

.face-dialog-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px 20px;
  border-bottom: 1px solid #f0f2f5;
}

.face-dialog-header h3 {
  margin: 0;
  font-size: 16px;
}

.close-btn {
  background: none;
  border: none;
  font-size: 18px;
  cursor: pointer;
  color: #999;
}

.face-dialog-body {
  padding: 20px;
}

.upload-section {
  margin-bottom: 20px;
  padding: 16px;
  background: #f9fafb;
  border-radius: 8px;
}

.upload-section p {
  margin: 0 0 10px 0;
  font-size: 13px;
  color: #666;
}

.upload-actions {
  display: flex;
  gap: 10px;
}

.btn-upload {
  background: #f0f0f0;
  border: 1px solid #d9d9d9;
  padding: 8px 16px;
  border-radius: 6px;
  cursor: pointer;
}

.btn-upload-confirm {
  background: #1677ff;
  color: #fff;
  border: none;
  padding: 8px 16px;
  border-radius: 6px;
  cursor: pointer;
}

.btn-upload-confirm:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.upload-tip {
  margin: 8px 0 0 0 !important;
  color: #1677ff !important;
  font-size: 12px !important;
}

.photo-list h4 {
  margin: 0 0 12px 0;
  font-size: 14px;
  color: #333;
}

.no-photo {
  color: #c0c4cc;
  text-align: center;
  padding: 20px;
}

.photo-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 12px;
}

.photo-item {
  position: relative;
  border: 1px solid #e5e6eb;
  border-radius: 8px;
  overflow: hidden;
}

.photo-img {
  width: 100%;
  height: 120px;
  object-fit: cover;
  display: block;
}

.photo-delete {
  position: absolute;
  top: 4px;
  right: 4px;
  background: rgba(255, 255, 255, 0.9);
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
  padding: 2px 4px;
}

.refresh-section {
  margin-top: 20px;
  padding: 16px;
  background: #f9fafb;
  border-radius: 8px;
  text-align: center;
}

.btn-refresh {
  background: #52c41a;
  color: #fff;
  border: none;
  padding: 10px 24px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 14px;
  font-weight: 600;
}

.btn-refresh:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.refresh-tip {
  margin: 10px 0 0 0;
  font-size: 12px;
  color: #999;
}
</style>