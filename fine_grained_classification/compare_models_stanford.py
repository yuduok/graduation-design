"""
细粒度分类对比评估脚本 - Stanford Cars
对比 CLIP、CoOp、CoCoOp 和 DynamicPrompt 在细粒度分类任务上的表现
"""
import os
import sys
import argparse
import json
import numpy as np
# 可选导入 matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
import torch
import torch.nn as nn
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
COOP_PATH = os.path.join(PROJECT_ROOT, "CoOp")
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, COOP_PATH)

import clip
from clip import clip


STANFORD_CARS_CLASSNAMES = [
    '1992 Acura Integra', '1992 Alfa Romeo 164', '1992 Chevrolet C/K 1500 Extended Cab',
    '1992 Dodge Dakota Club Cab', '1992 Ferrari F40', '1992 Ford E-Series Van',
    '1992 GMC Sierra Classic 1500 Extended Cab', '1992 Honda Prelude Si', '1992 Isuzu Rodeo',
    '1992 Mitsubishi Mirage', '1992 Plymouth Laser', '1992 Subaru Justy', '1992 Toyota MR2',
    '1992 Toyota Previa', '1992 Volkswagen Golf', '1993 Acura Integra GS-R', '1993 Chevrolet Camaro',
    '1993 Dodge Viper RT/10', '1993 Ford Mustang SVT Cobra R', '1993 Honda Civic EX',
    '1993 Mazda RX-7', '1993 Toyota Camry DX', '1993 Toyota Supra', '1994 Acura Integra',
    '1994 Chevrolet Corvette', '1994 Dodge Caravan', '1994 Ford Probe', '1994 Honda Civic CX',
    '1994 Mazda 626', '1994 Nissan Altima', '1994 Nissan Pickup', '1994 Toyota Celica GT',
    '1994 Toyota Celica GT-S', '1994 Toyota Supra', '1995 Acura NSX', '1995 Chevrolet Camaro Z28',
    '1995 Dodge Stealth R/T', '1995 Ford F-150 SuperCab', '1995 Honda Accord EX',
    '1995 Mazda MX-5 Miata', '1995 Mercedes-Benz 300-Class', '1995 Nissan Sentra SE-R',
    '1995 Porsche 911 Carrera 2', '1995 Toyota 4Runner', '1995 Toyota T100 Xtracab',
    '1996 Acura Integra RS', '1996 BMW Z3', '1996 Chevrolet Corvette Coupe',
    '1996 Dodge Stealth', '1996 Ferrari F355 Spider', '1996 Ford Mustang GT',
    '1996 Honda Accord Wagon', '1996 Honda Accord EX', '1996 Infiniti Q45', '1996 Jeep Wrangler',
    '1996 Lexus SC 300', '1996 Mazda Miata MX-5 Miata', '1996 Mercedes-Benz 300-Class',
    '1996 Mercury Villager', '1996 Nissan Maxima', '1996 Oldsmobile Achieva',
    '1996 Porsche 911 Turbo', '1996 Saturn SL2', '1996 Toyota Avalon', '1996 Toyota Celica GT-Four',
    '1996 Toyota Corolla', '1996 Volkswagen Golf', '1997 Acura Integra GS-R', '1997 BMW 3 Series',
    '1997 BMW 5 Series', '1997 Chevrolet Corvette Grand Sport', '1997 Chevrolet Tahoe',
    '1997 Ferrari F50', '1997 Honda Prelude SH', '1997 Infiniti J30', '1997 Jaguar XK8',
    '1997 Jeep Grand Cherokee', '1997 Lincoln Town Car', '1997 Mazda MPV', '1997 Mercedes-Benz C-Class',
    '1997 Mercedes-Benz E-Class', '1997 Mercedes-Benz S-Class', '1997 Mercury Mountaineer',
    '1997 Nissan 240SX', '1997 Nissan Pickup', '1997 Oldsmobile Silhouette',
    '1997 Plymouth Prowler', '1997 Pontiac Grand Prix', '1997 Porsche 911 Carrera',
    '1997 Rolls-Royce Silver Spur', '1997 Subaru Impreza WRX', '1997 Toyota Camry',
    '1997 Toyota Corolla', '1997 Volvo 240', '1998 Acura TL', '1998 BMW 3 Series Coupe',
    '1998 BMW 5 Series', '1998 Chevrolet S-10', '1998 Chevrolet S-10 Extended Cab',
    '1998 Chevrolet Silverado', '1998 Chrysler Sebring Convertible', '1998 Dodge Dakota Club Cab',
    '1998 Dodge Ram 1500 Club Cab', '1998 Dodge Ram Pickup', '1998 Ferrari F355 Berlinetta',
    '1998 Ford F-150', '1998 Ford F-150 Regular Cab', '1998 Ford F-150 SuperCab',
    '1998 Ford Expedition', '1998 Ford Explorer', '1998 Ford Mustang', '1998 Ford Mustang Cobra',
    '1998 Ford Ranger', '1998 Ford Ranger SuperCab', '1998 GMC Sonoma', '1998 GMC Sonoma Club Coupe',
    '1998 Honda Accord LX', '1998 Honda Accord', '1998 Honda Civic CX', '1998 Honda Civic EX',
    '1998 Honda Passport', '1998 Infiniti QX4', '1998 Isuzu Rodeo Sport', '1998 Isuzu Hombre',
    '1998 Jaguar XJ8', '1998 Jeep Wrangler', '1998 Kia Sephia', '1998 Lamborghini Diablo SV',
    '1998 Lincoln Navigator', '1998 Mazda 626 ES', '1998 Mazda MPV', '1998 Mazda Protege',
    '1998 Mazda RX-7', '1998 Mercedes-Benz C-Class', '1998 Mercedes-Benz CLK-Class',
    '1998 Mercury Mystique', '1998 Mitsubishi 3000GT', '1998 Mitsubishi Eclipse',
    '1998 Nissan Frontier', '1998 Nissan Maxima', '1998 Nissan Pathfinder',
    '1998 Nissan Sentra', '1998 Nissan Xterra', '1998 Oldsmobile Cutlass Supreme',
    '1998 Porsche 911 GT2', '1998 Subaru Forester', '1998 Subaru Legacy Outback',
    '1998 Suzuki Vitara', '1998 Toyota 4Runner', '1998 Toyota Camry', '1998 Toyota Camry Solara',
    '1998 Toyota Corolla', '1998 Toyota Sienna', '1998 Volkswagen Golf GTI', '1998 Volkswagen Jetta',
    '1998 Volvo C70', '1998 Volvo S40', '1999 Acura Integra GS', '1999 BMW 3 Series',
    '1999 BMW 5 Series', '1999 BMW M Coupe', '1999 BMW M5', '1999 BMW Z3',
    '1999 Cadillac DeVille', '1999 Chevrolet Camaro', '1999 Chevrolet Corvette', '1999 Chevrolet S-10',
    '1999 Chevrolet Silverado 1500', '1999 Chevrolet Silverado 1500 Extended Cab',
    '1999 Chevrolet Suburban 1500', '1999 Chevrolet Suburban 2500', '1999 Chevrolet Tahoe',
    '1999 Chevrolet Tracker', '1999 Chrysler 300M', '1999 Chrysler Cirrus', '1999 Chrysler Sebring',
    '1999 Chrysler Sebring Convertible', '1999 Chrysler Town & Country', '1999 Dodge Durango',
    '1999 Dodge Intrepid', '1999 Dodge Neon', '1999 Dodge Ram 1500', '1999 Dodge Ram 1500 Club Cab',
    '1999 Dodge Ram 2500', '1999 Dodge Ram 3500', '1999 Dodge Viper GTS', '1999 Dodge Viper RT/10',
    '1999 Eagle Talon', '1999 Ferrari 360 Modena', '1999 Ford Contour SVT', '1999 Ford Crown Victoria',
    '1999 Ford Escort', '1999 Ford Escort ZX2', '1999 Ford F-150', '1999 Ford F-150 Lightning',
    '1999 Ford F-150 SuperCab', '1999 Ford Explorer', '1999 Ford Mustang', '1999 Ford Mustang Cobra',
    '1999 Ford Mustang Cobra R', '1999 Ford Mustang GT', '1999 Ford Ranger', '1999 Ford Taurus',
    '1999 Ford Windstar', '1999 GMC Jimmy', '1999 GMC Sonoma', '1999 GMC Yukon',
    '1999 Honda Accord Coupe', '1999 Honda Civic EX', '1999 Honda Civic Si', '1999 Honda Odyssey',
    '1999 Honda Passport', '1999 Honda Prelude', '1999 Hyundai Elantra', '1999 Hyundai Sonata',
    '1999 Infiniti I30', '1999 Isuzu Amigo', '1999 Isuzu Rodeo', '1999 Isuzu Trooper',
    '1999 Jaguar XJ8', '1999 Jaguar XJR', '1999 Jaguar XK8', '1999 Jeep Cherokee',
    '1999 Jeep Grand Cherokee', '1999 Jeep Wrangler', '1999 Kia Sephia', '1999 Land Rover Discovery',
    '1999 Land Rover Freelander', '1999 Lexus ES 300', '1999 Lincoln Town Car', '1999 Lincoln Navigator',
    '1999 Mazda 626', '1999 Mazda B-Series', '1999 Mazda Miata MX-5 Miata', '1999 Mazda Millenia',
    '1999 Mazda MPV', '1999 Mazda Protege', '1999 Mazda RX-7', '1999 Mercury Cougar',
    '1999 Mercury Grand Marquis', '1999 Mercury Marauder', '1999 Mercury Mountaineer', '1999 Mitsubishi 3000GT',
    '1999 Mitsubishi Eclipse', '1999 Mitsubishi Galant', '1999 Nissan 240SX', '1999 Nissan Altima',
    '1999 Nissan Frontier', '1999 Nissan Maxima', '1999 Nissan Pathfinder', '1999 Nissan Pickup',
    '1999 Nissan Sentra', '1999 Nissan Xterra', '1999 Oldsmobile Alero', '1999 Oldsmobile Aurora',
    '1999 Oldsmobile Intrigue', '1999 Oldsmobile Silhouette', '1999 Plymouth Breeze',
    '1999 Plymouth Prowler', '1999 Pontiac Bonneville', '1999 Pontiac Firebird',
    '1999 Pontiac Grand Am', '1999 Pontiac Grand Prix', '1999 Pontiac Montana', '1999 Pontiac Sunfire',
    '1999 Porsche 911 Carrera', '1999 Porsche 911 Carrera 4', '1999 Porsche 911 Carrera 4 Cabriolet',
    '1999 Porsche 911 GT3', '1999 Porsche Boxster', '1999 Subaru Forester', '1999 Subaru Impreza',
    '1999 Subaru Impreza WRX', '1999 Subaru Legacy', '1999 Suzuki Esteem', '1999 Suzuki Grand Vitara',
    '1999 Suzuki Vitara', '1999 Toyota 4Runner', '1999 Toyota Avalon', '1999 Toyota Camry',
    '1999 Toyota Camry Solara', '1999 Toyota Celica', '1999 Toyota Corolla', '1999 Toyota Land Cruiser',
    '1999 Toyota MR2 Spyder', '1999 Toyota Sienna', '1999 Toyota Solara', '1999 Toyota Supra',
    '1999 Toyota Tundra', '1999 Volkswagen Cabrio', '1999 Volkswagen Golf', '1999 Volkswagen Jetta',
    '1999 Volkswagen Passat', '1999 Volvo C70', '1999 Volvo S70', '1999 Volvo V70',
    '1999 Volvo XC70', '2000 Acura Integra Type R', '2000 Acura NSX', '2000 Acura TL',
    '2000 Audi A4', '2000 Audi A6', '2000 Audi TT', '2000 BMW 3 Series',
    '2000 BMW 3 Series Coupe', '2000 BMW 5 Series', '2000 BMW 7 Series', '2000 BMW M3',
    '2000 BMW M5', '2000 BMW Z3', '2000 BMW Z8', '2000 Cadillac DeVille',
    '2000 Chevrolet Camaro', '2000 Chevrolet Corvette', '2000 Chevrolet Corvette Z06',
    '2000 Chevrolet Impala', '2000 Chevrolet Monte Carlo', '2000 Chevrolet S-10',
    '2000 Chevrolet Silverado', '2000 Chevrolet Suburban', '2000 Chevrolet Tahoe',
    '2000 Chevrolet Tracker', '2000 Chrysler 300M', '2000 Chrysler Cirrus', '2000 Chrysler Concorde',
    '2000 Chrysler Sebring', '2000 Chrysler Sebring Convertible', '2000 Chrysler Town & Country',
    '2000 Dodge Avenger', '2000 Dodge Caravan', '2000 Dodge Dakota', '2000 Dodge Durango',
    '2000 Dodge Grand Caravan', '2000 Dodge Intrepid', '2000 Dodge Neon', '2000 Dodge Ram Pickup',
    '2000 Dodge Ram Van', '2000 Dodge Stealth', '2000 Dodge Stratus', '2000 Dodge Viper',
    '2000 Ford Contour SVT', '2000 Ford Crown Victoria', '2000 Ford Escort', '2000 Ford Expedition',
    '2000 Ford Explorer', '2000 Ford F-150', '2000 Ford F-150 Lightning', '2000 Ford F-150 SuperCab',
    '2000 Ford F-150 SuperCrew', '2000 Ford F-250', '2000 Ford F-350', '2000 Ford Mustang',
    '2000 Ford Mustang Cobra', '2000 Ford Mustang GT', '2000 Ford Ranger', '2000 Ford Ranger SuperCab',
    '2000 Ford Taurus', '2000 Ford Windstar', '2000 Ford Windstar LX', '2000 Ford Windstar SE',
    '2000 Ford Windstar SEL', '2000 GMC Jimmy', '2000 GMC Savana', '2000 GMC Sierra',
    '2000 GMC Sonoma', '2000 GMC Yukon', '2000 Honda Accord', '2000 Honda Civic',
    '2000 Honda Civic CX', '2000 Honda Civic EX', '2000 Honda Civic Si', '2000 Honda Odyssey',
    '2000 Honda Passport', '2000 Honda Prelude', '2000 Hyundai Accent', '2000 Hyundai Elantra',
    '2000 Hyundai Sonata', '2000 Hyundai Tiburon', '2000 Infiniti G20', '2000 Infiniti I30',
    '2000 Infiniti Q45', '2000 Infiniti QX4', '2000 Isuzu Amigo', '2000 Isuzu Rodeo',
    '2000 Isuzu Trooper', '2000 Jaguar S-Type', '2000 Jaguar XJ8', '2000 Jaguar XJR',
    '2000 Jaguar XK8', '2000 Jeep Cherokee', '2000 Jeep Grand Cherokee', '2000 Jeep Wrangler',
    '2000 Kia Sephia', '2000 Kia Sportage', '2000 Lamborghini Diablo', '2000 Lamborghini Diablo SV',
    '2000 Land Rover Discovery', '2000 Land Rover Freelander', '2000 Land Rover Range Rover',
    '2000 Lexus ES 300', '2000 Lexus GS 300', '2000 Lexus GS 400', '2000 Lexus LS 400',
    '2000 Lincoln LS', '2000 Lincoln Navigator', '2000 Lincoln Town Car', '2000 Mazda 626',
    '2000 Mazda B-Series', '2000 Mazda Millenia', '2000 Mazda Miata MX-5 Miata', '2000 Mazda MPV',
    '2000 Mazda Protege', '2000 Mazda RX-7', '2000 Mercury Cougar', '2000 Mercury Grand Marquis',
    '2000 Mercury Marauder', '2000 Mercury Mountaineer', '2000 Mercury Sable', '2000 Mitsubishi 3000GT',
    '2000 Mitsubishi Eclipse', '2000 Mitsubishi Galant', '2000 Mitsubishi Mirage', '2000 Mitsubishi Montero',
    '2000 Mitsubishi Montero Sport', '2000 Nissan Altima', '2000 Nissan Frontier', '2000 Nissan Maxima',
    '2000 Nissan Pathfinder', '2000 Nissan Pickup', '2000 Nissan Sentra', '2000 Nissan Xterra',
    '2000 Oldsmobile Alero', '2000 Oldsmobile Aurora', '2000 Oldsmobile Bravada', '2000 Oldsmobile Intrigue',
    '2000 Oldsmobile Silhouette', '2000 Plymouth Breeze', '2000 Plymouth Prowler', '2000 Pontiac Aztek',
    '2000 Pontiac Bonneville', '2000 Pontiac Firebird', '2000 Pontiac Grand Am', '2000 Pontiac Grand Prix',
    '2000 Pontiac Montana', '2000 Pontiac Sunfire', '2000 Pontiac Trans Sport', '2000 Porsche 911 Carrera',
    '2000 Porsche 911 Carrera 4', '2000 Porsche 911 Carrera Cabriolet', '2000 Porsche 911 GT3',
    '2000 Porsche Boxster', '2000 Subaru Forester', '2000 Subaru Impreza', '2000 Subaru Impreza WRX',
    '2000 Subaru Legacy', '2000 Subaru Outback', '2000 Suzuki Esteem', '2000 Suzuki Grand Vitara',
    '2000 Suzuki Vitara', '2000 Toyota 4Runner', '2000 Toyota Avalon', '2000 Toyota Camry',
    '2000 Toyota Camry Solara', '2000 Toyota Celica', '2000 Toyota Celica GT-S', '2000 Toyota Corolla',
    '2000 Toyota Echo', '2000 Toyota Land Cruiser', '2000 Toyota MR2 Spyder', '2000 Toyota Prius',
    '2000 Toyota RAV4', '2000 Toyota Sequoia', '2000 Toyota Sienna', '2000 Toyota Solara',
    '2000 Toyota Supra', '2000 Toyota Tundra', '2000 Toyota Tundra Access Cab', '2000 Volkswagen Cabrio',
    '2000 Volkswagen Golf', '2000 Volkswagen Jetta', '2000 Volkswagen Passat', '2000 Volvo C70',
    '2000 Volvo S40', '2000 Volvo S70', '2000 Volvo V40', '2000 Volvo V70', '2000 Volvo XC70',
    '2001 Acura Integra', '2001 Acura NSX', '2001 Acura RL', '2001 Acura TL', '2001 Audi A4',
    '2001 Audi A6', '2001 Audi A8', '2001 Audi Allroad', '2001 Audi TT', '2001 BMW 3 Series',
    '2001 BMW 3 Series Coupe', '2001 BMW 5 Series', '2001 BMW 7 Series', '2001 BMW M3',
    '2001 BMW M5', '2001 BMW X5', '2001 BMW Z3', '2001 BMW Z8', '2001 Cadillac Catera',
    '2001 Cadillac DeVille', '2001 Cadillac Eldorado', '2001 Cadillac Seville',
    '2001 Chevrolet Camaro', '2001 Chevrolet Corvette', '2001 Chevrolet Corvette Z06',
    '2001 Chevrolet Impala', '2001 Chevrolet Monte Carlo', '2001 Chevrolet S-10', '2001 Chevrolet Silverado',
    '2001 Chevrolet Suburban', '2001 Chevrolet Tahoe', '2001 Chevrolet Tracker', '2001 Chrysler 300M',
    '2001 Chrysler Concorde', '2001 Chrysler PT Cruiser', '2001 Chrysler Sebring', '2001 Chrysler Sebring Convertible',
    '2001 Chrysler Town & Country', '2001 Dodge Avenger', '2001 Dodge Caravan', '2001 Dodge Dakota',
    '2001 Dodge Durango', '2001 Dodge Grand Caravan', '2001 Dodge Intrepid', '2001 Dodge Neon',
    '2001 Dodge Ram Pickup', '2001 Dodge Ram Van', '2001 Dodge Stratus', '2001 Dodge Viper',
    '2001 Ford Crown Victoria', '2001 Ford Escort', '2001 Ford Expedition', '2001 Ford Explorer',
    '2001 Ford F-150', '2001 Ford F-150 Lightning', '2001 Ford F-150 SuperCab', '2001 Ford F-150 SuperCrew',
    '2001 Ford F-250', '2001 Ford F-350', '2001 Ford Mustang', '2001 Ford Mustang Cobra',
    '2001 Ford Mustang GT', '2001 Ford Ranger', '2001 Ford Ranger SuperCab', '2001 Ford Taurus',
    '2001 Ford Windstar', '2001 GMC Jimmy', '2001 GMC Savana', '2001 GMC Sierra', '2001 GMC Sonoma',
    '2001 GMC Yukon', '2001 GMC Yukon XL', '2001 Honda Accord', '2001 Honda Civic', '2001 Honda Civic Si',
    '2001 Honda Odyssey', '2001 Honda Passport', '2001 Honda Prelude', '2001 Honda S2000',
    '2001 Hyundai Accent', '2001 Hyundai Elantra', '2001 Hyundai Sonata', '2001 Hyundai Tiburon',
    '2001 Hyundai XG300', '2001 Infiniti G20', '2001 Infiniti I30', '2001 Infiniti QX4',
    '2000 Isuzu Amigo', '2000 Isuzu Rodeo', '2000 Isuzu Trooper', '2000 Jaguar S-Type', '2000 Jaguar XJ8',
    '2000 Jaguar XJR', '2000 Jaguar XK8', '2000 Jeep Cherokee', '2000 Jeep Grand Cherokee',
    '2000 Jeep Wrangler', '2000 Kia Sephia', '2000 Kia Sportage', '2000 Lamborghini Diablo',
    '2000 Lamborghini Diablo SV', '2000 Land Rover Discovery', '2000 Land Rover Freelander',
    '2000 Land Rover Range Rover', '2000 Lexus ES 300', '2000 Lexus GS 300', '2000 Lexus GS 400',
    '2000 Lexus LS 400', '2000 Lincoln LS', '2000 Lincoln Navigator', '2000 Lincoln Town Car',
    '2000 Mazda 626', '2000 Mazda B-Series', '2000 Mazda Millenia', '2000 Mazda Miata MX-5 Miata',
    '2000 Mazda MPV', '2000 Mazda Protege', '2000 Mazda RX-7', '2000 Mercury Cougar', '2000 Mercury Grand Marquis',
    '2000 Mercury Marauder', '2000 Mercury Mountaineer', '2000 Mercury Sable', '2000 Mitsubishi 3000GT',
    '2000 Mitsubishi Eclipse', '2000 Mitsubishi Galant', '2000 Mitsubishi Mirage', '2000 Mitsubishi Montero',
    '2000 Mitsubishi Montero Sport', '2000 Nissan Altima', '2000 Nissan Frontier', '2000 Nissan Maxima',
    '2000 Nissan Pathfinder', '2000 Nissan Pickup', '2000 Nissan Sentra', '2000 Nissan Xterra',
    '2000 Oldsmobile Alero', '2000 Oldsmobile Aurora', '2000 Oldsmobile Bravada', '2000 Oldsmobile Intrigue',
    '2000 Oldsmobile Silhouette', '2000 Plymouth Breeze', '2000 Plymouth Prowler', '2000 Pontiac Aztek',
    '2000 Pontiac Bonneville', '2000 Pontiac Firebird', '2000 Pontiac Grand Am', '2000 Pontiac Grand Prix',
    '2000 Pontiac Montana', '2000 Pontiac Sunfire', '2000 Pontiac Trans Sport', '2000 Porsche 911 Carrera',
    '2000 Porsche 911 Carrera 4', '2000 Porsche 911 Carrera Cabriolet', '2000 Porsche 911 GT3',
    '2000 Porsche Boxster', '2000 Subaru Forester', '2000 Subaru Impreza', '2000 Subaru Impreza WRX',
    '2000 Subaru Legacy', '2000 Subaru Outback', '2000 Suzuki Esteem', '2000 Suzuki Grand Vitara',
    '2000 Suzuki Vitara', '2000 Toyota 4Runner', '2000 Toyota Avalon', '2000 Toyota Camry',
    '2000 Toyota Camry Solara', '2000 Toyota Celica', '2000 Toyota Celica GT-S', '2000 Toyota Corolla',
    '2000 Toyota Echo', '2000 Toyota Land Cruiser', '2000 Toyota MR2 Spyder', '2000 Toyota Prius',
    '2000 Toyota RAV4', '2000 Toyota Sequoia', '2000 Toyota Sienna', '2000 Toyota Solara',
    '2000 Toyota Supra', '2000 Toyota Tundra', '2000 Toyota Tundra Access Cab', '2000 Volkswagen Cabrio',
    '2000 Volkswagen Golf', '2000 Volkswagen Jetta', '2000 Volkswagen Passat', '2000 Volvo C70',
    '2000 Volvo S40', '2000 Volvo S70', '2000 Volvo V40', '2000 Volvo V70', '2000 Volvo XC70'
]


class StanfordCarsDataset(Dataset):
    def __init__(self, data_root, split="test"):
        self.data_root = data_root
        self.dataset_dir = os.path.join(data_root, "stanford_cars")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_StanfordCars.json")

        if os.path.exists(self.split_path):
            with open(self.split_path, 'r') as f:
                splits = json.load(f)
            raw_data = splits.get(split, [])
            self.data = []
            for item in raw_data:
                if isinstance(item, list) and len(item) >= 2:
                    self.data.append({'image': item[0], 'label': int(item[1])})
                elif isinstance(item, dict):
                    self.data.append(item)
        else:
            raise FileNotFoundError(f"Split file not found: {self.split_path}")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711]
            )
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.dataset_dir, item['image'])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return {'img': image, 'label': item['label'], 'image_name': item['image']}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare fine-grained classification models - Stanford Cars")
    parser.add_argument("--dataset", type=str, default="stanford_cars",
                       choices=["oxford_pets", "stanford_cars"],
                       help="dataset name")
    parser.add_argument("--data-root", type=str,
                       default="/home/zengyule/graduation-design/data",
                       help="path to dataset root")
    parser.add_argument("--backbone", type=str, default="RN50",
                       choices=["RN50", "RN101", "ViT-B/16", "ViT-B/32"],
                       help="CLIP backbone")
    parser.add_argument("--batch-size", type=int, default=16,
                       help="batch size for evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="number of data workers")
    parser.add_argument("--output-dir", type=str, default="comparison_results_stanford",
                       help="output directory")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu", "mps"],
                       help="device to use")
    parser.add_argument("--models", nargs="+",
                       default=["clip", "coop", "cocoop", "dynamic"],
                       help="models to compare")
    parser.add_argument("--model-dirs", type=str, default=None,
                       help="JSON string mapping model names to checkpoint directories")
    return parser.parse_args()


class ZeroShotCLIP:
    def __init__(self, backbone, classnames, device):
        self.device = device
        self.classnames = classnames
        print(f"Loading Zero-shot CLIP ({backbone})...")
        self.clip_model, _ = clip.load(backbone, device=device)
        self.clip_model.eval()

        prompts = [f"a photo of a {name.replace('_', ' ')}" for name in classnames]
        self.tokens = clip.tokenize(prompts).to(device)

    @torch.no_grad()
    def evaluate(self, dataloader):
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for batch in tqdm(dataloader, desc="Zero-shot CLIP"):
            images = batch['img'].to(self.device)
            labels = batch['label']

            image_features = self.clip_model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text_features = self.clip_model.encode_text(self.tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

            logits = self.clip_model.logit_scale.exp() * image_features @ text_features.T
            probs = logits.softmax(dim=1)
            preds = logits.argmax(dim=1)

            correct += (preds == labels.to(self.device)).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.append(probs.cpu())

        all_probs = torch.cat(all_probs)
        accuracy = correct / total

        return {
            "accuracy": accuracy,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs
        }


def load_clip_model(backbone, device):
    print(f"Loading CLIP ({backbone})...")
    clip_model, _ = clip.load(backbone, device=device)
    clip_model.eval()
    clip_model.float()
    return clip_model


class CoOpClassifier(nn.Module):
    def __init__(self, clip_model, classnames, ctx, tokenized_prompts, token_prefix, token_suffix):
        super().__init__()
        self.clip_model = clip_model
        self.classnames = classnames
        self.ctx = ctx
        self.n_ctx = ctx.shape[0]
        self.tokenized_prompts = tokenized_prompts
        self.token_prefix = token_prefix
        self.token_suffix = token_suffix
        self.dtype = clip_model.dtype

    def forward(self, images):
        image_features = self.clip_model.encode_image(images.type(self.dtype))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        prompts = []
        for i in range(len(self.classnames)):
            prefix = self.token_prefix[i:i+1]
            ctx_i = self.ctx.unsqueeze(0)
            suffix = self.token_suffix[i:i+1]
            prompt = torch.cat([prefix, ctx_i, suffix], dim=1)
            prompts.append(prompt)
        prompts = torch.cat(prompts, dim=0)

        x = prompts + self.clip_model.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.clip_model.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.clip_model.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection

        text_features = x / x.norm(dim=-1, keepdim=True)
        logit_scale = self.clip_model.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits


class CoOpEvaluator:
    def __init__(self, backbone, checkpoint_dir, classnames, device):
        self.device = device
        self.clip_model = load_clip_model(backbone, device)

        print(f"Loading CoOp model from {checkpoint_dir}...")
        checkpoint_path = os.path.join(checkpoint_dir, "prompt_learner", "model-best.pth.tar")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_dir, "prompt_learner", "model.pth.tar-20")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_dir, "prompt_learner", "model.pth.tar")

        if not os.path.exists(checkpoint_path):
            print(f"Warning: CoOp checkpoint not found at {checkpoint_path}")
            self.model = None
            return

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint.get('epoch', 'N/A')
        print(f"Loaded CoOp checkpoint from epoch {epoch}")

        n_ctx = state_dict['ctx'].shape[0]
        ctx = state_dict['ctx'].to(device)

        from clip.simple_tokenizer import SimpleTokenizer
        _tokenizer = SimpleTokenizer()
        prompts = [f"a photo of a {name.replace('_', ' ')}." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.clip_model.dtype)

        token_prefix = embedding[:, :1, :]
        token_suffix = embedding[:, 1 + n_ctx:, :]

        self.model = CoOpClassifier(self.clip_model, classnames, ctx, tokenized_prompts, token_prefix, token_suffix)
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, dataloader):
        if self.model is None:
            return {"accuracy": 0, "predictions": [], "labels": [], "probabilities": torch.zeros(1)}

        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for batch in tqdm(dataloader, desc="CoOp"):
            images = batch['img'].to(self.device)
            labels = batch['label']

            logits = self.model(images)
            probs = logits.softmax(dim=1)
            preds = logits.argmax(dim=1)

            correct += (preds == labels.to(self.device)).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.append(probs.cpu())

        all_probs = torch.cat(all_probs)
        accuracy = correct / total

        return {
            "accuracy": accuracy,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs
        }


class SoftPromptAdapter(nn.Module):
    def __init__(self, vis_dim=512, ctx_dim=512):
        super().__init__()
        hidden_dim = vis_dim // 16
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(vis_dim, hidden_dim)),
            ("relu", nn.ReLU(inplace=True)),
            ("linear2", nn.Linear(hidden_dim, ctx_dim))
        ]))

    def forward(self, image_features, base_ctx, n_cls):
        batch_size = image_features.shape[0]
        bias = self.meta_net(image_features).unsqueeze(1)
        ctx = base_ctx.unsqueeze(0).expand(batch_size, -1, -1)
        ctx_shifted = ctx + bias
        ctx_shifted = ctx_shifted.unsqueeze(1).expand(-1, n_cls, -1, -1)
        return ctx_shifted


class DynamicPromptClassifier(nn.Module):
    def __init__(self, clip_model, classnames, ctx, class_adaptive_factors, meta_net_state_dict, tokenized_prompts, token_prefix, token_suffix):
        super().__init__()
        self.clip_model = clip_model
        self.classnames = classnames
        self.ctx = nn.Parameter(ctx)
        self.class_adaptive_factors = nn.Parameter(class_adaptive_factors.squeeze(0))
        self.n_ctx = ctx.shape[0]
        self.tokenized_prompts = tokenized_prompts
        self.token_prefix = token_prefix
        self.token_suffix = token_suffix
        self.dtype = clip_model.dtype

        vis_dim = clip_model.visual.output_dim
        ctx_dim = ctx.shape[-1]

        adapter = SoftPromptAdapter(vis_dim=vis_dim, ctx_dim=ctx_dim)
        if meta_net_state_dict is not None:
            adapter.meta_net.load_state_dict(meta_net_state_dict)
        self.soft_prompt_adapter = adapter

    def forward(self, images):
        image_features = self.clip_model.encode_image(images.type(self.dtype))
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)

        base_ctx = self.ctx * self.class_adaptive_factors
        ctx_shifted = self.soft_prompt_adapter(image_features_norm, base_ctx, len(self.classnames))

        batch_size = images.shape[0]
        prefix = self.token_prefix.unsqueeze(0).expand(batch_size, -1, -1, -1)
        suffix = self.token_suffix.unsqueeze(0).expand(batch_size, -1, -1, -1)

        prompts = torch.cat([prefix, ctx_shifted, suffix], dim=2)

        logits_list = []
        logit_scale = self.clip_model.logit_scale.exp()
        for i in range(batch_size):
            prompt_i = prompts[i]
            x = prompt_i + self.clip_model.positional_embedding.type(self.dtype)
            x = x.permute(1, 0, 2)
            x = self.clip_model.transformer(x)
            x = x.permute(1, 0, 2)
            x = self.clip_model.ln_final(x).type(self.dtype)
            x = x[torch.arange(x.shape[0]), self.tokenized_prompts.argmax(dim=-1)] @ self.clip_model.text_projection
            text_features = x / x.norm(dim=-1, keepdim=True)
            logit = logit_scale * image_features_norm[i] @ text_features.t()
            logits_list.append(logit)

        logits = torch.stack(logits_list)
        return logits


class DynamicPromptEvaluator:
    def __init__(self, backbone, checkpoint_dir, classnames, device):
        self.device = device
        self.clip_model = load_clip_model(backbone, device)

        print(f"Loading DynamicPrompt model from {checkpoint_dir}...")
        checkpoint_path = os.path.join(checkpoint_dir, "prompt_learner", "model-best.pth.tar")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_dir, "prompt_learner", "model.pth.tar-20")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_dir, "prompt_learner", "model.pth.tar")

        if not os.path.exists(checkpoint_path):
            print(f"Warning: DynamicPrompt checkpoint not found at {checkpoint_path}")
            self.model = None
            return

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint.get('epoch', 'N/A')
        print(f"Loaded DynamicPrompt checkpoint from epoch {epoch}")
        print(f"Available state_dict keys: {list(state_dict.keys())[:10]}...")

        n_ctx = state_dict['ctx'].shape[0]
        ctx = state_dict['ctx'].to(device)

        meta_net_state_dict = None
        soft_prompt_keys = [k for k in state_dict.keys() if 'soft_prompt_adapter' in k or 'meta_net' in k]
        if soft_prompt_keys:
            print(f"Soft prompt related keys: {soft_prompt_keys}")
            if 'soft_prompt_adapter.meta_net.linear1.weight' in state_dict:
                # DynamicPrompt: 去掉 'soft_prompt_adapter.meta_net.' 前缀
                meta_net_state_dict = {
                    'linear1.weight': state_dict['soft_prompt_adapter.meta_net.linear1.weight'],
                    'linear1.bias': state_dict['soft_prompt_adapter.meta_net.linear1.bias'],
                    'linear2.weight': state_dict['soft_prompt_adapter.meta_net.linear2.weight'],
                    'linear2.bias': state_dict['soft_prompt_adapter.meta_net.linear2.bias'],
                }
            elif 'meta_net.linear1.weight' in state_dict:
                # CoCoOp: 去掉 'meta_net.' 前缀
                meta_net_state_dict = {
                    'linear1.weight': state_dict['meta_net.linear1.weight'],
                    'linear1.bias': state_dict['meta_net.linear1.bias'],
                    'linear2.weight': state_dict['meta_net.linear2.weight'],
                    'linear2.bias': state_dict['meta_net.linear2.bias'],
                }

        class_adaptive_factors = None
        if 'class_adaptive_factors' in state_dict:
            class_adaptive_factors = state_dict['class_adaptive_factors'].to(device)
        else:
            class_adaptive_factors = torch.ones(1, n_ctx, ctx.shape[-1], device=device)

        from clip.simple_tokenizer import SimpleTokenizer
        _tokenizer = SimpleTokenizer()
        prompts = [f"a photo of a {name.replace('_', ' ')}." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.clip_model.dtype)

        token_prefix = embedding[:, :1, :]
        token_suffix = embedding[:, 1 + n_ctx:, :]

        self.model = DynamicPromptClassifier(
            self.clip_model, classnames, ctx, class_adaptive_factors,
            meta_net_state_dict, tokenized_prompts, token_prefix, token_suffix
        )
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self, dataloader):
        if self.model is None:
            return {"accuracy": 0, "predictions": [], "labels": [], "probabilities": torch.zeros(1)}

        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        all_probs = []

        for batch in tqdm(dataloader, desc="DynamicPrompt"):
            images = batch['img'].to(self.device)
            labels = batch['label']

            logits = self.model(images)
            probs = logits.softmax(dim=1)
            preds = logits.argmax(dim=1)

            correct += (preds == labels.to(self.device)).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.append(probs.cpu())

        all_probs = torch.cat(all_probs)
        accuracy = correct / total

        return {
            "accuracy": accuracy,
            "predictions": all_preds,
            "labels": all_labels,
            "probabilities": all_probs
        }


class CoCoOpEvaluator(DynamicPromptEvaluator):
    def __init__(self, backbone, checkpoint_dir, classnames, device):
        self.device = device
        self.clip_model = load_clip_model(backbone, device)

        print(f"Loading CoCoOp model from {checkpoint_dir}...")
        checkpoint_path = os.path.join(checkpoint_dir, "prompt_learner", "model-best.pth.tar")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_dir, "prompt_learner", "model.pth.tar-20")
        if not os.path.exists(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_dir, "prompt_learner", "model.pth.tar")

        if not os.path.exists(checkpoint_path):
            print(f"Warning: CoCoOp checkpoint not found at {checkpoint_path}")
            self.model = None
            return

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        state_dict = checkpoint["state_dict"]
        epoch = checkpoint.get('epoch', 'N/A')
        print(f"Loaded CoCoOp checkpoint from epoch {epoch}")

        n_ctx = state_dict['ctx'].shape[0]
        ctx = state_dict['ctx'].to(device)

        meta_net_state_dict = None
        if 'meta_net.linear1.weight' in state_dict:
            # CoCoOp: 去掉前缀 'meta_net.'
            meta_net_state_dict = {
                'linear1.weight': state_dict['meta_net.linear1.weight'],
                'linear1.bias': state_dict['meta_net.linear1.bias'],
                'linear2.weight': state_dict['meta_net.linear2.weight'],
                'linear2.bias': state_dict['meta_net.linear2.bias'],
            }

        class_adaptive_factors = torch.ones(1, n_ctx, ctx.shape[-1], device=device)

        from clip.simple_tokenizer import SimpleTokenizer
        _tokenizer = SimpleTokenizer()
        prompts = [f"a photo of a {name.replace('_', ' ')}." for name in classnames]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)

        with torch.no_grad():
            embedding = self.clip_model.token_embedding(tokenized_prompts).type(self.clip_model.dtype)

        token_prefix = embedding[:, :1, :]
        token_suffix = embedding[:, 1 + n_ctx:, :]

        self.model = DynamicPromptClassifier(
            self.clip_model, classnames, ctx, class_adaptive_factors,
            meta_net_state_dict, tokenized_prompts, token_prefix, token_suffix
        )
        self.model.to(device)
        self.model.eval()


def compute_top_k_accuracy(probs, labels, k=5):
    topk_preds = torch.topk(probs, k=k, dim=1).indices
    correct = 0
    for i, label in enumerate(labels):
        if label in topk_preds[i]:
            correct += 1
    return correct / len(labels)


def plot_accuracy_comparison(results, output_dir):
    models = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in models]

    plt.figure(figsize=(10, 6))
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    bars = plt.bar(models, accuracies, color=colors[:len(models)])

    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Stanford Cars Fine-Grained Classification Accuracy Comparison", fontsize=14)
    plt.ylim(0, 100)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved accuracy comparison to {os.path.join(output_dir, 'accuracy_comparison.png')}")


def plot_confidence_distribution(results, output_dir):
    plt.figure(figsize=(12, 6))

    for i, (model_name, result) in enumerate(results.items()):
        probs = result["probabilities"]
        if probs.shape[0] == 1:
            continue
        max_probs = probs.max(dim=1).values.numpy()

        plt.subplot(1, len(results), i + 1)
        plt.hist(max_probs, bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel("Prediction Confidence")
        plt.ylabel("Count")
        plt.title(f"{model_name}\nMean: {max_probs.mean():.3f}")
        plt.xlim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confidence_distribution.png"), dpi=150)
    plt.close()
    print(f"Saved confidence distribution to {os.path.join(output_dir, 'confidence_distribution.png')}")


def plot_top_k_accuracies(results, output_dir, k_values=[1, 3, 5]):
    valid_results = {k: v for k, v in results.items() if v["probabilities"].shape[0] > 1}
    if not valid_results:
        print("No valid results for Top-K comparison")
        return

    labels = list(valid_results.values())[0]["labels"]
    labels_tensor = torch.tensor(labels)

    topk_results = {}
    for model_name, result in valid_results.items():
        probs = result["probabilities"]
        topk_accs = {}
        for k in k_values:
            topk_accs[k] = compute_top_k_accuracy(probs, labels_tensor, k) * 100
        topk_results[model_name] = topk_accs

    x = np.arange(len(k_values))
    width = 0.2
    num_models = len(topk_results)

    plt.figure(figsize=(12, 6))
    for i, (model_name, topk_accs) in enumerate(topk_results.items()):
        accuracies = [topk_accs[k] for k in k_values]
        offset = (i - num_models/2 + 0.5) * width
        bars = plt.bar(x + offset, accuracies, width, label=model_name)
        for bar, acc in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=8)

    plt.xlabel("Top-K")
    plt.ylabel("Accuracy (%)")
    plt.title("Stanford Cars Top-K Accuracy Comparison")
    plt.xticks(x, [f"Top-{k}" for k in k_values])
    plt.legend()
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top_k_accuracies.png"), dpi=150)
    plt.close()
    print(f"Saved Top-K accuracies to {os.path.join(output_dir, 'top_k_accuracies.png')}")


def print_detailed_results(results):
    print("\n" + "="*70)
    print("DETAILED RESULTS - Stanford Cars")
    print("="*70)

    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy: {result['accuracy']*100:.2f}%")

        probs = result["probabilities"]
        if probs.shape[0] == 1:
            continue

        labels = result["labels"]
        preds = result["predictions"]

        max_probs = probs.max(dim=1).values
        print(f"  Mean Confidence: {max_probs.mean()*100:.2f}%")
        print(f"  Median Confidence: {max_probs.median()*100:.2f}%")

        correct_mask = torch.tensor(preds) == torch.tensor(labels)
        incorrect_mask = ~correct_mask

        if correct_mask.any():
            correct_confs = max_probs[correct_mask]
            print(f"  Correct Prediction Confidence: {correct_confs.mean()*100:.2f}%")
        if incorrect_mask.any():
            incorrect_confs = max_probs[incorrect_mask]
            print(f"  Incorrect Prediction Confidence: {incorrect_confs.mean()*100:.2f}%")


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = "cpu"

    print("="*60)
    print("FINE-GRAINED CLASSIFICATION MODEL COMPARISON - STANFORD CARS")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Backbone: {args.backbone}")
    print(f"Device: {device}")
    print(f"Models to compare: {args.models}")
    print("="*60)

    classnames = STANFORD_CARS_CLASSNAMES

    print("\nLoading test dataset...")
    dataset = StanfordCarsDataset(args.data_root, split="test")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    print(f"Test samples: {len(dataset)}")

    results = {}

    model_dirs = {}
    if args.model_dirs:
        model_dirs = json.loads(args.model_dirs)

    if "clip" in args.models:
        print("\n" + "-"*40)
        print("Evaluating Zero-shot CLIP")
        print("-"*40)
        evaluator = ZeroShotCLIP(args.backbone, classnames, device)
        results["Zero-shot CLIP"] = evaluator.evaluate(dataloader)

    if "coop" in args.models:
        print("\n" + "-"*40)
        print("Evaluating CoOp")
        print("-"*40)
        coop_dir = model_dirs.get("coop", f"{PROJECT_ROOT}/fine_grained_classification/output_fgd/{args.dataset}/CoOp/shots_16/seed_1")
        evaluator = CoOpEvaluator(args.backbone, coop_dir, classnames, device)
        results["CoOp"] = evaluator.evaluate(dataloader)

    if "cocoop" in args.models:
        print("\n" + "-"*40)
        print("Evaluating CoCoOp")
        print("-"*40)
        cocoop_dir = model_dirs.get("cocoop", f"{PROJECT_ROOT}/fine_grained_classification/output_fgd/{args.dataset}/CoCoOp/shots_16/seed_1")
        evaluator = CoCoOpEvaluator(args.backbone, cocoop_dir, classnames, device)
        results["CoCoOp"] = evaluator.evaluate(dataloader)

    if "dynamic" in args.models:
        print("\n" + "-"*40)
        print("Evaluating DynamicPrompt")
        print("-"*40)
        dynamic_dir = model_dirs.get("dynamic", f"{PROJECT_ROOT}/fine_grained_classification/output_fgd/{args.dataset}/DynamicPromptTrainer/shots_16/seed_1")
        evaluator = DynamicPromptEvaluator(args.backbone, dynamic_dir, classnames, device)
        results["DynamicPrompt"] = evaluator.evaluate(dataloader)

    print("\n" + "="*60)
    print("SUMMARY - STANFORD CARS")
    print("="*60)
    print(f"{'Model':<20} {'Accuracy':<15}")
    print("-"*35)
    for model_name, result in results.items():
        print(f"{model_name:<20} {result['accuracy']*100:.2f}%")
    print("="*60)

    print_detailed_results(results)

    print("\nGenerating visualization plots...")
    plot_accuracy_comparison(results, args.output_dir)
    plot_confidence_distribution(results, args.output_dir)
    plot_top_k_accuracies(results, args.output_dir)

    summary = {
        "dataset": args.dataset,
        "backbone": args.backbone,
        "models": {},
        "best_model": max(results.keys(), key=lambda k: results[k]["accuracy"])
    }

    for model_name, result in results.items():
        summary["models"][model_name] = {
            "accuracy": result["accuracy"],
            "mean_confidence": result["probabilities"].max(dim=1).values.mean().item() if result["probabilities"].shape[0] > 1 else 0
        }

    summary_path = os.path.join(args.output_dir, "comparison_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    print("\nComparison complete!")


if __name__ == "__main__":
    main()
