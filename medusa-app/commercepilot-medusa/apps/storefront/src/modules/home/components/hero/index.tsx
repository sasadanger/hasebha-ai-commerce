import { Button, Heading, Text } from "@medusajs/ui"
import LocalizedClientLink from "@modules/common/components/localized-client-link"

const Hero = () => {
  return (
    <div className="h-[75vh] w-full border-b border-ui-border-base relative bg-ui-bg-subtle">
      <div className="absolute inset-0 z-10 flex flex-col justify-center items-center text-center small:p-32 gap-6">
        <span>
          <Heading
            level="h1"
            className="text-4xl leading-10 text-ui-fg-base font-normal"
          >
            HASEBHA
          </Heading>
          <Heading
            level="h2"
            className="text-2xl leading-10 text-ui-fg-subtle font-normal"
            dir="rtl"
            lang="ar"
          >
            حاسبها
          </Heading>
        </span>
        <Heading
          level="h2"
          className="text-xl leading-8 text-ui-fg-subtle font-normal"
        >
          AI-Powered Commerce Intelligence
        </Heading>
        <Text
          className="text-ui-fg-subtle max-w-lg text-lg"
          dir="rtl"
          lang="ar"
        >
          الداتا تحسبها… وإنت تقرر.
        </Text>
        <Text className="text-ui-fg-subtle max-w-lg">
          Every order placed here is analyzed in real time by HASEBHA&apos;s
          AI decision engine.
        </Text>
        <LocalizedClientLink href="/store">
          <Button variant="primary">Shop all products</Button>
        </LocalizedClientLink>
      </div>
    </div>
  )
}

export default Hero
