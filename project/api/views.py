from django.shortcuts import render
from rest_framework.viewsets import ModelViewSet
from .models import Item
from .serializers import ItemSerializer
import json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import subprocess, os

class ItemViewSet(ModelViewSet):
    queryset = Item.objects.all()
    serializer_class = ItemSerializer

@csrf_exempt  # CSRF 검사를 비활성화 (테스트용)
def run_init(request):
    if request.method == 'GET':
        json_file_path = '/home/ubuntu/ID/init.json'
        if os.path.exists(json_file_path):
            with open(json_file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)  # json.load 사용
            return JsonResponse(data, safe=False)  # safe=False로 리스트 지원
        else:
            return JsonResponse({"error": "init.json file not found"}, status=404)
    else:
        return JsonResponse({"error": "Invalid request method"}, status=405)



@csrf_exempt
def run_top_script(request):
    if request.method == "POST":
        try:
            # 요청 데이터 파싱
            data = json.loads(request.body)
            start_date = data.get("start_date")
            end_date = data.get("end_date")
            combination = ",".join(data.get("combination", []))  # 리스트를 콤마로 연결
            buy_signal = ",".join(data.get("buy_signal", []))
            sell_signal = ",".join(data.get("sell_signal", []))
            # 입력 검증
            if not start_date or not end_date or not combination:
                return JsonResponse({"error": "Invalid input data"}, status=400)

            # 스크립트 경로
            script_path = "/home/ubuntu/ID/top_response.py"  

            # 스크립트 실행
            result = subprocess.run(
                [
                    "python3", script_path,
                    "--start_date", start_date,
                    "--end_date", end_date,
                    "--combination", combination,
                    "--buy_signal", buy_signal,
                    "--sell_signal", sell_signal
                ],
                capture_output=True,
                text=True
            )

            # 스크립트 실행 결과 확인
            if result.returncode != 0:
                return JsonResponse({
                    "error": "Script execution failed",
                    "stderr": result.stderr,
                    "stdout": result.stdout
                }, status=500)

            # 스크립트 출력(JSON 형식)을 읽어 반환
            output = json.loads(result.stdout.strip())
            return JsonResponse(output)

        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON in script output"}, status=500)
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)

    return JsonResponse({"error": "Invalid request method"}, status=405)

@csrf_exempt
def run_script(request):
    if request.method == "POST":
        try:
            # 요청 데이터 파싱
            data = json.loads(request.body)
            start_date = data.get("start_date")
            end_date = data.get("end_date")
            combination = ",".join(data.get("combination", []))  # 리스트를 콤마로 연결
            asset = data.get("asset")

            # 입력 검증
            if not start_date or not end_date or not combination:
                return JsonResponse({"error": "Invalid input data"}, status=400)

            # 스크립트 경로
            script_path = "/home/ubuntu/ID/backtesting/response.py"

            # 스크립트 실행
            result = subprocess.run(
                [
                    "python3", script_path,
                    "--start_date", start_date,
                    "--end_date", end_date,
                    "--combination", combination,
                    "--asset", str(asset)
                ],
                capture_output=True,
                text=True,
                check=True,
                cwd="/home/ubuntu/ID/backtesting"
            )

            # 스크립트 출력(JSON 형식)을 읽어 반환
            output = result.stdout.strip()

            # 디버깅 로그 추가
            print(f"Script output: {output}")

            # JSON 변환
            response_data = json.loads(output)  # 스크립트가 JSON 형식으로 출력한다고 가정
            return JsonResponse(response_data)

        except subprocess.CalledProcessError as e:
            return JsonResponse({
                "error": f"Script execution failed: {e}",
                "stderr": e.stderr.strip(),
                "stdout": e.stdout.strip()
            }, status=500)

        except json.JSONDecodeError as e:
            return JsonResponse({
                "error": f"Invalid JSON output from script: {e}",
                "script_output": output
            }, status=500)

        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
