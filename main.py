from common_config import config


def main():
    if config.model == 'GDGAN':
        from GDGAN.gdgan import GDGAN
        from GDGAN.gdgan_config import validate_args
        gan = GDGAN(validate_args(config))

    gan.run()


if __name__ == '__main__':
    main()
