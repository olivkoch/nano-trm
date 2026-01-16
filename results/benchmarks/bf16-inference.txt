Failed to import adam2
--- Benchmarking on CUDA ---
--- Precision: BFLOAT16 ---
Manually casting TRM to torch.bfloat16
Manually casting TRM to torch.bfloat16
Manually casting TRM to torch.bfloat16
--------------------------------------------------------------------------------
Size: 32x32, Batch: 32, Precision: BF16
Model                | Params       | Throughput (samples/s)    | Mode      
--------------------------------------------------------------------------------
Basic CNN            | 8.90 M       | 109834.47                 | Autocast  
ResNet18             | 11.30 M      | 66941.51                  | Autocast  
ResNet50             | 24.03 M      | 31649.15                  | Autocast  
EfficientNet-B2      | 8.06 M       | 21833.96                  | Autocast  
DiT-Medium           | 2.76 M       | 1255.26                   | Autocast  
TRM-Light            | 230 K        | 27638.43                  | Native    
TRM-Medium           | 1.44 M       | 2690.43                   | Native    
TRM-Heavy            | 9.18 M       | 254.98                    | Native    
--------------------------------------------------------------------------------
Manually casting TRM to torch.bfloat16
Manually casting TRM to torch.bfloat16
Manually casting TRM to torch.bfloat16
--------------------------------------------------------------------------------
Size: 46x46, Batch: 32, Precision: BF16
Model                | Params       | Throughput (samples/s)    | Mode      
--------------------------------------------------------------------------------
Basic CNN            | 8.90 M       | 51925.77                  | Autocast  
ResNet18             | 11.30 M      | 37767.38                  | Autocast  
ResNet50             | 24.03 M      | 15708.64                  | Autocast  
EfficientNet-B2      | 8.06 M       | 10200.31                  | Autocast  
DiT-Medium           | 3.04 M       | 517.66                    | Autocast  
TRM-Light            | 230 K        | 11887.35                  | Native    
TRM-Medium           | 1.44 M       | 1133.47                   | Native    
TRM-Heavy            | 9.18 M       | 94.01                     | Native    
--------------------------------------------------------------------------------
Manually casting TRM to torch.bfloat16
Manually casting TRM to torch.bfloat16
Manually casting TRM to torch.bfloat16
--------------------------------------------------------------------------------
Size: 64x64, Batch: 32, Precision: BF16
Model                | Params       | Throughput (samples/s)    | Mode      
--------------------------------------------------------------------------------
Basic CNN            | 8.90 M       | 38424.77                  | Autocast  
ResNet18             | 11.30 M      | 33616.62                  | Autocast  
ResNet50             | 24.03 M      | 13722.57                  | Autocast  
EfficientNet-B2      | 8.06 M       | 9477.85                   | Autocast  
DiT-Medium           | 3.55 M       | 223.18                    | Autocast  
TRM-Light            | 230 K        | 3936.44                   | Native    
TRM-Medium           | 1.44 M       | 435.26                    | Native    
TRM-Heavy            | 9.18 M       | 37.39                     | Native    
--------------------------------------------------------------------------------
Manually casting TRM to torch.bfloat16
Manually casting TRM to torch.bfloat16
Manually casting TRM to torch.bfloat16
--------------------------------------------------------------------------------
Size: 90x90, Batch: 32, Precision: BF16
Model                | Params       | Throughput (samples/s)    | Mode      
--------------------------------------------------------------------------------
Basic CNN            | 8.90 M       | 32357.22                  | Autocast  
ResNet18             | 11.30 M      | 33227.14                  | Autocast  
ResNet50             | 24.03 M      | 13326.62                  | Autocast  
EfficientNet-B2      | 8.06 M       | 9259.84                   | Autocast  
DiT-Medium           | 4.57 M       | 75.10                     | Autocast  
TRM-Light            | 230 K        | 1448.95                   | Native    
TRM-Medium           | 1.44 M       | 124.96                    | Native    
TRM-Heavy            | 9.18 M       | 11.27                     | Native    
--------------------------------------------------------------------------------
Manually casting TRM to torch.bfloat16
Manually casting TRM to torch.bfloat16
Manually casting TRM to torch.bfloat16
--------------------------------------------------------------------------------
Size: 128x128, Batch: 32, Precision: BF16
Model                | Params       | Throughput (samples/s)    | Mode      
--------------------------------------------------------------------------------
Basic CNN            | 8.90 M       | 18840.75                  | Autocast  
ResNet18             | 11.30 M      | 23934.11                  | Autocast  
ResNet50             | 24.03 M      | 9667.22                   | Autocast  
EfficientNet-B2      | 8.06 M       | 6807.97                   | Autocast  
DiT-Medium           | 6.69 M       | 23.38                     | Autocast  
TRM-Light            | 230 K        | 421.92                    | Native    
TRM-Medium           | 1.44 M       | 35.62                     | Native    
TRM-Heavy            | 9.18 M       | 3.38                      | Native    
--------------------------------------------------------------------------------
