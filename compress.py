from imports import get_orig_model, compress_model, get_model_size

if __name__ == "__main__":
    print("Загрузка моделей")
    orig_model, orig_tokenizer = get_orig_model()
    comp_model, comp_tokenizer = compress_model()

    comp_size = get_model_size(comp_model)
    orig_size = get_model_size(orig_model)
    compression = orig_size / comp_size
    print(f'Вес исходной модели: {orig_size}. Вес сжатой модели: {comp_size}. Сжатие: {compression}')

    orig_model.save_pretrained("Models/Qwen3-8B-full")
    orig_tokenizer.save_pretrained("Models/Qwen3-8B-full")
    comp_model.save_pretrained("Models/Qwen3-8B-4bit")
    comp_tokenizer.save_pretrained("Models/Qwen3-8B-4bit")