cd ../../

#python train_end2end_detector.py  configurations/EMBER/training/ResNet18/resnet18_ember_configuration_file_pos_weight_0.875.json
#python evaluate_end2end_detector.py  configurations/EMBER/validation/ResNet18/resnet18_ember_validation_set_configuration_file_pos_weight_0.875.json
python evaluate_end2end_detector.py  configurations/EMBER/test/ResNet18/resnet18_ember_test_set_configuration_file_pos_weight_0.875.json