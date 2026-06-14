# ai
## Dataset Processing

본 프로젝트는 AI Hub HTP(House-Tree-Person) 데이터셋을 기반으로 구축되었다.
House, Tree, Person 도메인으로 분리하여 각각 독립적인 객체 탐지 모델 학습이 가능하도록 재구성하였다.

특히 Person 데이터셋의 경우 원본 클래스 구조를 검토한 후, 성별 구분 및 신발 유형 분리 등 일부 클래스를 재정의하여 학습에 활용하였다.

### House Dataset

* 14 Classes

### Tree Dataset

* 14 Classes

### Person Dataset

* 20 Classes

  * male_person
  * female_person
  * sneakers
  * male_shoes
  * female_shoes
  * 기타 신체 부위 클래스

