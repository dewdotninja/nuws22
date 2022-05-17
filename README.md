## การฝึกอบรมเชิงปฏิบัติการเรื่อง "โครงข่ายประสาทเทียมและการเรียนรู้เชิงลึก"

<p />  
<p align="center">
<img src="https://drive.google.com/uc?id=1gfBnit4FRu-MvGjOGpVEw21TxwzNbq-R" width=500 />
</p>

<img align=left src="https://i.imgur.com/CzEUVpd.jpg" width=200 /> 
<ul>
  <li /><b>ชื่อเรื่อง :</b> โครงข่ายประสาทเทียมและการเรียนรู้เชิงลึก (Artificial Neural Networks and Deep Learning)
  <li /><b>วิทยากร :</b> ดร.วโรดม ตู้จินดา
  <li /><b>สถานที่ :</b> ภาควิชาฟิสิกส์ คณะวิทยาศาสตร์ มหาวิทยาลัยนเรศวร
  <li /><b>วัน-เวลา :</b> 14 - 15 พฤษภาคม 2565
</ul>
<hr>
จากคำบรรยายในวิกิพิเดีย ปัญญาประดิษฐ์ (artificial intelligence) 
หรือที่นิยมเรียกโดยย่อว่า AI คือเชาว์ปัญญาที่แสดงให้เห็นในเครื่องจักรกล 
แตกต่างจากปัญญาธรรมชาติจากสมองของมนุษย์หรือสัตว์ ดังนั้น AI จะครอบคลุมนวัตกรรมทั้งหมดที่ทำให้คอมพิวเตอร์มีความชาญฉลาดเข้าสู่ปัญญาของมนุษย์ 
ซึ่งมีขอบเขตที่กว้างมาก การเรียนรู้ของเครื่อง (machine learning) คือเซตย่อยของเอไอที่เน้นการศึกษาขั้นตอนวิธีทางคอมพิวเตอร์ในการเรียนรู้และปรับตัวจากข้อมูล ซึ่งสามารถทำได้
หลายแนวทางเช่นการหาค่าเหมาะที่สุด ทฤษฏีกราฟ แต่ในการฝึกอบรมครั้งนี้จะเน้นการใช้โครงข่ายประสาทเทียม (artificial neural network) ซึ่งต่อไปจะอ้างถึงโดยตัวย่อ 
ANN ในการเรียนรู้ กล่าวได้ว่า ANN คือระบบคอมพิวเตอร์หรือโมเดลทางคณิตศาสตร์ที่ใช้ในการจำลองสมองทางชีวภาพ โดยอาศัยการเรียนรู้หรือเรียกว่าการฝึก (training) 
โดยข้อมูลตัวอย่าง เมื่อเขียนภาพรวมความสัมพันธ์ของขอบเขตปัญหาที่กล่าวมาจะได้ดังแสดงในรูป 

<p />  
<p align="center">
<img src="https://drive.google.com/uc?id=1r_T5zq9MMcGXptpF_k1BU5nfC4Kq6WM_" width=400 />
</p>
<div align="center">ความสัมพันธ์ของปัญญาประดิษฐ์ การเรียนรู้ของเครื่อง และการเรียนรู้เชิงลึก</div>

### โครงข่ายประสาทเทียมและการเรียนรู้เชิงลึก

มักมีความเข้าใจผิดกับคำว่า "เชิงลึก" ในการเรียนรู้เชิงลึก ซึ่งมิได้มีความหมายในเชิงปรัชญา หรือแปลว่าการเรียนรู้อย่างถ่องแท้แต่อย่างใด แต่เป็นคำคุณศัพท์ที่ขยายความ เกี่ยวกับจำนวนชั้นของโครงข่ายประสาทเทียม โดย ANN ที่มีเฉพาะชั้นที่เป็นอินพุตและเอาต์พุตเรียกว่าเป็นแบบตื้น (shallow) ส่วนโครงข่ายประสาทเทียมเชิงลึก (deep neural network) ซึ่งต่อไปจะเรียกโดยย่อว่า DNN นอกจากชั้นอินพุตและเอาต์พุตแล้วยังประกอบด้วยชั้นแฝง (hidden layer) อย่างน้อยหนึ่งชั้น 

สถาปัตยกรรมของโครงข่ายประสาทเทียมที่ใช้ในการเรียนรู้เชิงลึกมีจำแนกได้เป็นหลายประเภทขึ้นกับการใช้งาน ในการฝึกอบรมครั้งนี้จะศึกษาตั้งแต่โมเดล DNN ขั้นพื้นฐาน 
หลักการของการเรียนรู้โดยปรับค่าเกรเดียนต์ของค่าสูญเสียในทิศทางเข้าสู่ค่าต่ำสุด ปัญหาการฟิตเกิน (overfitting) และการแก้ไข หลักการของตัวหาค่าเหมาะที่สุด 
(optimizers) ประเภทต่างๆ โมเดล CNN (Convolutional Neural Networks) สำหรับงานที่เกี่ยวข้องกับข้อมูลภาพ การถ่ายโอนการเรียนรู้ (transfer 
learning) และโมเดล RNN (Recurrent Neural Networks), GRU (Gated Recurrent Unit) , LSTM (Long Short Term Memory) สำหรับข้อมูลลำดับ เช่นงานด้าน NLP (Natural Language Processing) การพยากรณ์ข้อมูลฐานเวลา 

<p />  
<p align="center">
<img src="https://drive.google.com/uc?id=1ccOQjyevWY2bErjs-nIevtGjCVlpHNwA" width=600 />
</p>

<hr>
### กำหนดการ

#### อังคาร 14 พฤษภาคม 2565


#### พุธ 15 พฤษภาคม 2565


<hr>
### การดาวน์โหลด .ZIP

ผู้เข้าอบรมหรือผู้สนใจทั่วไปสามารถดาวน์โหลดไฟล์ทั้งหมดในหน้านี้ได้โดยคลิกที่ปุ่ม [Code] สีเขียวด้านบนขวา และเลือก Download ZIP ดังแสดงในภาพ

<img src="https://drive.google.com/uc?id=1MN-ZsN0TtzqcV5ad1hPrz3cvInyrSg7o" width=500 />


### อุปกรณ์ที่ใช้ในการอบรม

<ul>
  <li />เครื่องคอมพิวเตอร์ระบบปฏิบัติการ Windows หรือ Mac-OSX (ซอฟต์แวร์ที่ใช้สามารถใช้งานฟรีทั้งหมด ควรติดตั้งก่อนการอบรม)
  <li />สัญญาณ WiFi เพื่อเชื่อมต่ออินเทอร์เน็ต 
</ul>

### ซอฟต์แวร์และการติดตั้ง


